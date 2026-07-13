import math

import numpy as np
import taichi as ti


@ti.data_oriented
class SinglePhaseUGKWP2D:
    """Isothermal 2D UGKWP solver for a periodic shear wave.

    Conserved variables are [rho, rho*u, rho*v].  The flux solver uses
    the UGKWP wave-particle split from the integral BGK solution.  The
    collisional part is evolved by the analytic gas-kinetic wave flux, while
    the no-collision probability exp(-dt/tau) part is represented by
    simulation particles and contributes a finite-volume crossing flux.

    The benchmark-facing rho/u initialization, diagnostics, and main loop are
    kept compatible with GKS.py.  This file intentionally changes the step
    procedure and its helper kernels rather than the surrounding framework.
    """

    def __init__(self, nx=256, ny=256, particles_per_cell=300):
        self.nx = nx
        self.ny = ny
        self.particles_per_cell = particles_per_cell
        self.max_particles = self.nx * self.ny * self.particles_per_cell
        self.dt = 0.01
        self.cs = 1.0 / math.sqrt(3.0)
        self.cs2 = self.cs * self.cs
        self.nu = 0.00001
        self.mach = 0.3
        self.tau = self.nu / self.cs2

        self.rho = ti.field(ti.f32, shape=(self.nx, self.ny))
        self.rho_next = ti.field(ti.f32, shape=(self.nx, self.ny))
        self.u = ti.Vector.field(2, ti.f32, shape=(self.nx, self.ny))
        self.u_next = ti.Vector.field(2, ti.f32, shape=(self.nx, self.ny))
        self.face_x = ti.Vector.field(3, ti.f32, shape=(self.nx, self.ny))
        self.face_y = ti.Vector.field(3, ti.f32, shape=(self.nx, self.ny))
        self.p_x = ti.field(ti.f32, shape=self.max_particles)
        self.p_y = ti.field(ti.f32, shape=self.max_particles)
        self.p_vx = ti.field(ti.f32, shape=self.max_particles)
        self.p_vy = ti.field(ti.f32, shape=self.max_particles)
        self.p_mass = ti.field(ti.f32, shape=self.max_particles)
        self.p_tfree = ti.field(ti.f32, shape=self.max_particles)

    @ti.func
    def wrap_i(self, i):
        return (i + self.nx) % self.nx

    @ti.func
    def wrap_j(self, j):
        return (j + self.ny) % self.ny

    @ti.func
    def minmod(self, a, b):
        out = 0.0
        if a * b > 0.0:
            out = a
            if ti.abs(b) < ti.abs(a):
                out = b
        return out

    @ti.func
    def limited_scalar_slope(self, qm, q0, qp):
        return self.minmod(q0 - qm, qp - q0)

    @ti.func
    def limited_vector_slope(self, qm, q0, qp):
        return ti.Vector(
            [
                self.limited_scalar_slope(qm[0], q0[0], qp[0]),
                self.limited_scalar_slope(qm[1], q0[1], qp[1]),
            ]
        )

    @ti.func
    def particle_index(self, i, j, p):
        return (i * self.ny + j) * self.particles_per_cell + p

    @ti.func
    def normal_pair(self):
        r1 = ti.max(ti.random(ti.f32), 1.0e-6)
        r2 = ti.random(ti.f32)
        mag = ti.sqrt(-2.0 * ti.log(r1))
        ang = 2.0 * math.pi * r2
        return mag * ti.cos(ang), mag * ti.sin(ang)

    @ti.func
    def erf_approx(self, x):
        ax = ti.abs(x)
        t = 1.0 / (1.0 + 0.3275911 * ax)
        poly = (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t)
        y = 1.0 - poly * ti.exp(-ax * ax)
        if x < 0.0:
            y = -y
        return y

    @ti.func
    def full_moments_1d(self, u):
        cs2 = ti.static(self.cs2)
        cs4 = ti.static(self.cs2 * self.cs2)
        u2 = u * u
        out = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        out[0] = 1.0
        out[1] = u
        out[2] = u2 + cs2
        out[3] = u * u2 + 3.0 * cs2 * u
        out[4] = u2 * u2 + 6.0 * cs2 * u2 + 3.0 * cs4
        return out

    @ti.func
    def half_moments_normal(self, u):
        """Return int_{xi>0} xi^k N(u, cs^2) dxi, k=0..4."""
        cs = ti.static(self.cs)
        cs2 = ti.static(self.cs2)
        inv_sqrt2 = ti.static(1.0 / math.sqrt(2.0))
        inv_sqrt2pi = ti.static(1.0 / math.sqrt(2.0 * math.pi))

        z = u / cs
        phi = ti.exp(-0.5 * z * z) * inv_sqrt2pi
        hp = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        hp[0] = 0.5 * (1.0 + self.erf_approx(z * inv_sqrt2))
        hp[1] = u * hp[0] + cs * phi
        for n in ti.static(range(2, 5)):
            hp[n] = u * hp[n - 1] + (n - 1) * cs2 * hp[n - 2]
        return hp

    @ti.func
    def moment2(self, mx, my, ix, iy):
        return mx[ix] * my[iy]

    @ti.func
    def solve_coefficients(self, h, vel):
        cs2_inv = ti.static(1.0 / self.cs2)
        a = ti.Vector([0.0, 0.0, 0.0])
        a[1] = cs2_inv * (h[1] - vel[0] * h[0])
        a[2] = cs2_inv * (h[2] - vel[1] * h[0])
        a[0] = h[0] - vel[0] * a[1] - vel[1] * a[2]
        return a

    @ti.func
    def spatial_coefficients(self, rho, vel, grad_rho, grad_u):
        cs2_inv = ti.static(1.0 / self.cs2)
        a = ti.Vector([0.0, 0.0, 0.0])
        b = ti.Vector([0.0, 0.0, 0.0])

        ux_x = grad_u[0, 0]
        uy_x = grad_u[1, 0]
        ux_y = grad_u[0, 1]
        uy_y = grad_u[1, 1]

        a[0] = grad_rho[0] - rho * (vel[0] * ux_x + vel[1] * uy_x) * cs2_inv
        a[1] = rho * ux_x * cs2_inv
        a[2] = rho * uy_x * cs2_inv

        b[0] = grad_rho[1] - rho * (vel[0] * ux_y + vel[1] * uy_y) * cs2_inv
        b[1] = rho * ux_y * cs2_inv
        b[2] = rho * uy_y * cs2_inv
        return a, b

    @ti.func
    def temporal_coefficients(self, vel, mx, my, a, b):
        h = ti.Vector([0.0, 0.0, 0.0])
        for c in ti.static(range(3)):
            rx = 0
            ry = 0
            if c == 1:
                rx = 1
            elif c == 2:
                ry = 1

            s = (
                a[0] * self.moment2(mx, my, 1 + rx, ry)
                + b[0] * self.moment2(mx, my, rx, 1 + ry)
                + a[1] * self.moment2(mx, my, 2 + rx, ry)
                + b[2] * self.moment2(mx, my, rx, 2 + ry)
                + (a[2] + b[1]) * self.moment2(mx, my, 1 + rx, 1 + ry)
            )
            h[c] = -s
        return self.solve_coefficients(h, vel)

    @ti.func
    def full_flux_moment(self, mx, my, rho, a, b, A, component, kind):
        rx = 0
        ry = 0
        if component == 1:
            rx = 1
        elif component == 2:
            ry = 1

        out = 0.0
        if kind == 0:
            out = rho * self.moment2(mx, my, 1 + rx, ry)
        elif kind == 1:
            out = (
                a[0] * self.moment2(mx, my, 2 + rx, ry)
                + b[0] * self.moment2(mx, my, 1 + rx, 1 + ry)
                + a[1] * self.moment2(mx, my, 3 + rx, ry)
                + b[2] * self.moment2(mx, my, 1 + rx, 2 + ry)
                + (a[2] + b[1]) * self.moment2(mx, my, 2 + rx, 1 + ry)
            )
        else:
            out = (
                A[0] * self.moment2(mx, my, 1 + rx, ry)
                + A[1] * self.moment2(mx, my, 2 + rx, ry)
                + A[2] * self.moment2(mx, my, 1 + rx, 1 + ry)
            )
        return out

    @ti.func
    def half_flux_moment(self, hx, my, rho, a, b, A, component, kind):
        rx = 0
        ry = 0
        if component == 1:
            rx = 1
        elif component == 2:
            ry = 1

        out = 0.0
        if kind == 0:
            out = rho * self.moment2(hx, my, 1 + rx, ry)
        elif kind == 1:
            out = (
                a[0] * self.moment2(hx, my, 2 + rx, ry)
                + b[0] * self.moment2(hx, my, 1 + rx, 1 + ry)
                + a[1] * self.moment2(hx, my, 3 + rx, ry)
                + b[2] * self.moment2(hx, my, 1 + rx, 2 + ry)
                + (a[2] + b[1]) * self.moment2(hx, my, 2 + rx, 1 + ry)
            )
        else:
            out = (
                A[0] * self.moment2(hx, my, 1 + rx, ry)
                + A[1] * self.moment2(hx, my, 2 + rx, ry)
                + A[2] * self.moment2(hx, my, 1 + rx, 1 + ry)
            )
        return out

    @ti.func
    def interface_equilibrium(self, rho_l, mx_l, my_l, hp_l, rho_r, mx_r, my_r, hp_r):
        hn_r = mx_r - hp_r
        w = ti.Vector([0.0, 0.0, 0.0])
        w[0] = rho_l * hp_l[0] * my_l[0] + rho_r * hn_r[0] * my_r[0]
        w[1] = rho_l * hp_l[1] * my_l[0] + rho_r * hn_r[1] * my_r[0]
        w[2] = rho_l * hp_l[0] * my_l[1] + rho_r * hn_r[0] * my_r[1]
        rho0 = ti.max(w[0], 1.0e-8)
        vel0 = ti.Vector([w[1] / rho0, w[2] / rho0])
        return rho0, vel0

    @ti.func
    def gks_flux_local(self, rho_l, vel_l, grad_rho_l, grad_u_l, rho_r, vel_r, grad_rho_r, grad_u_r):
        mx_l = self.full_moments_1d(vel_l[0])
        my_l = self.full_moments_1d(vel_l[1])
        mx_r = self.full_moments_1d(vel_r[0])
        my_r = self.full_moments_1d(vel_r[1])
        hp_l = self.half_moments_normal(vel_l[0])
        hp_r = self.half_moments_normal(vel_r[0])
        hn_r = mx_r - hp_r

        a_l, b_l = self.spatial_coefficients(rho_l, vel_l, grad_rho_l, grad_u_l)
        a_r, b_r = self.spatial_coefficients(rho_r, vel_r, grad_rho_r, grad_u_r)
        A_l = self.temporal_coefficients(vel_l, mx_l, my_l, a_l, b_l)
        A_r = self.temporal_coefficients(vel_r, mx_r, my_r, a_r, b_r)

        rho0, vel0 = self.interface_equilibrium(rho_l, mx_l, my_l, hp_l, rho_r, mx_r, my_r, hp_r)
        mx0 = self.full_moments_1d(vel0[0])
        my0 = self.full_moments_1d(vel0[1])
        grad_rho0 = 0.5 * (grad_rho_l + grad_rho_r)
        grad_u0 = 0.5 * (grad_u_l + grad_u_r)
        a0, b0 = self.spatial_coefficients(rho0, vel0, grad_rho0, grad_u0)
        A0 = self.temporal_coefficients(vel0, mx0, my0, a0, b0)

        tau = ti.static(self.tau)
        dt = ti.static(self.dt)
        E = ti.exp(-dt / tau)
        c_eq = 1.0 - tau * (1.0 - E) / dt
        c_eq_space = (2.0 * tau * tau * (1.0 - E) - tau * dt * E - tau * dt) / dt
        c_eq_time = 0.5 * dt - tau + tau * tau * (1.0 - E) / dt
        c_init = tau * (1.0 - E) / dt
        c_init_space = -(2.0 * tau * tau * (1.0 - E) - tau * dt * E) / dt
        c_init_time = -tau * tau * (1.0 - E) / dt

        flux = ti.Vector([0.0, 0.0, 0.0])
        for component in ti.static(range(3)):
            eq = self.full_flux_moment(mx0, my0, rho0, a0, b0, A0, component, 0)
            eq_space = self.full_flux_moment(mx0, my0, rho0, a0, b0, A0, component, 1)
            eq_time = self.full_flux_moment(mx0, my0, rho0, a0, b0, A0, component, 2)

            init = (
                self.half_flux_moment(hp_l, my_l, rho_l, a_l, b_l, A_l, component, 0)
                + self.half_flux_moment(hn_r, my_r, rho_r, a_r, b_r, A_r, component, 0)
            )
            init_space = (
                self.half_flux_moment(hp_l, my_l, rho_l, a_l, b_l, A_l, component, 1)
                + self.half_flux_moment(hn_r, my_r, rho_r, a_r, b_r, A_r, component, 1)
            )
            init_time = (
                self.half_flux_moment(hp_l, my_l, rho_l, a_l, b_l, A_l, component, 2)
                + self.half_flux_moment(hn_r, my_r, rho_r, a_r, b_r, A_r, component, 2)
            )

            flux[component] = (
                c_eq * eq
                + c_eq_space * eq_space
                + c_eq_time * eq_time
                + c_init * init
                + c_init_space * init_space
                + c_init_time * init_time
            )
        return flux

    @ti.func
    def equilibrium_flux_local(self, rho_l, vel_l, grad_rho_l, grad_u_l, rho_r, vel_r, grad_rho_r, grad_u_r):
        mx_l = self.full_moments_1d(vel_l[0])
        my_l = self.full_moments_1d(vel_l[1])
        mx_r = self.full_moments_1d(vel_r[0])
        my_r = self.full_moments_1d(vel_r[1])
        hp_l = self.half_moments_normal(vel_l[0])
        hp_r = self.half_moments_normal(vel_r[0])

        rho0, vel0 = self.interface_equilibrium(rho_l, mx_l, my_l, hp_l, rho_r, mx_r, my_r, hp_r)
        mx0 = self.full_moments_1d(vel0[0])
        my0 = self.full_moments_1d(vel0[1])
        grad_rho0 = 0.5 * (grad_rho_l + grad_rho_r)
        grad_u0 = 0.5 * (grad_u_l + grad_u_r)
        a0, b0 = self.spatial_coefficients(rho0, vel0, grad_rho0, grad_u0)
        A0 = self.temporal_coefficients(vel0, mx0, my0, a0, b0)

        tau = ti.static(self.tau)
        dt = ti.static(self.dt)
        E = ti.exp(-dt / tau)
        c_eq = 1.0 - tau * (1.0 - E) / dt
        c_eq_space = (2.0 * tau * tau * (1.0 - E) - tau * dt * E - tau * dt) / dt
        c_eq_time = 0.5 * dt - tau + tau * tau * (1.0 - E) / dt

        flux = ti.Vector([0.0, 0.0, 0.0])
        for component in ti.static(range(3)):
            eq = self.full_flux_moment(mx0, my0, rho0, a0, b0, A0, component, 0)
            eq_space = self.full_flux_moment(mx0, my0, rho0, a0, b0, A0, component, 1)
            eq_time = self.full_flux_moment(mx0, my0, rho0, a0, b0, A0, component, 2)
            flux[component] = c_eq * eq + c_eq_space * eq_space + c_eq_time * eq_time
        return flux

    @ti.func
    def cell_gradients_world(self, i, j):
        ip = self.wrap_i(i + 1)
        im = self.wrap_i(i - 1)
        jp = self.wrap_j(j + 1)
        jm = self.wrap_j(j - 1)
        grad_rho = ti.Vector(
            [
                self.limited_scalar_slope(self.rho[im, j], self.rho[i, j], self.rho[ip, j]),
                self.limited_scalar_slope(self.rho[i, jm], self.rho[i, j], self.rho[i, jp]),
            ]
        )
        sx = self.limited_vector_slope(self.u[im, j], self.u[i, j], self.u[ip, j])
        sy = self.limited_vector_slope(self.u[i, jm], self.u[i, j], self.u[i, jp])
        grad_u = ti.Matrix.zero(ti.f32, 2, 2)
        grad_u[0, 0] = sx[0]
        grad_u[1, 0] = sx[1]
        grad_u[0, 1] = sy[0]
        grad_u[1, 1] = sy[1]
        return grad_rho, grad_u

    @ti.func
    def reconstruct_x_face(self, i_l, j_l, i_r, j_r):
        gr_l, gu_l = self.cell_gradients_world(i_l, j_l)
        gr_r, gu_r = self.cell_gradients_world(i_r, j_r)
        rho_l = ti.max(self.rho[i_l, j_l] + 0.5 * gr_l[0], 1.0e-8)
        rho_r = ti.max(self.rho[i_r, j_r] - 0.5 * gr_r[0], 1.0e-8)
        vel_l = self.u[i_l, j_l] + 0.5 * ti.Vector([gu_l[0, 0], gu_l[1, 0]])
        vel_r = self.u[i_r, j_r] - 0.5 * ti.Vector([gu_r[0, 0], gu_r[1, 0]])
        return rho_l, vel_l, gr_l, gu_l, rho_r, vel_r, gr_r, gu_r

    @ti.func
    def rotate_y_state(self, vel_w, grad_rho_w, grad_u_w):
        vel_l = ti.Vector([vel_w[1], vel_w[0]])
        grad_rho_l = ti.Vector([grad_rho_w[1], grad_rho_w[0]])
        grad_u_l = ti.Matrix.zero(ti.f32, 2, 2)
        grad_u_l[0, 0] = grad_u_w[1, 1]
        grad_u_l[1, 0] = grad_u_w[0, 1]
        grad_u_l[0, 1] = grad_u_w[1, 0]
        grad_u_l[1, 1] = grad_u_w[0, 0]
        return vel_l, grad_rho_l, grad_u_l

    @ti.kernel
    def init(self):
        for i, j in self.rho:
            y = (ti.cast(j, ti.f32) + 0.5) / ti.cast(self.ny, ti.f32)
            amp = self.mach * ti.static(self.cs)
            ux = amp * ti.sin(2.0 * math.pi * y)
            self.rho[i, j] = 1.0
            self.u[i, j] = ti.Vector([ux, 0.0])
            for p in range(self.particles_per_cell):
                k = self.particle_index(i, j, p)
                z0, z1 = self.normal_pair()
                self.p_x[k] = ti.cast(i, ti.f32) + ti.random(ti.f32)
                self.p_y[k] = ti.cast(j, ti.f32) + ti.random(ti.f32)
                self.p_vx[k] = ux + ti.static(self.cs) * z0
                self.p_vy[k] = ti.static(self.cs) * z1
                self.p_mass[k] = 1.0 / ti.cast(self.particles_per_cell, ti.f32)
                self.p_tfree[k] = -ti.static(self.tau) * ti.log(ti.max(ti.random(ti.f32), 1.0e-6))

    @ti.kernel
    def compute_face_fluxes_x(self):
        for i, j in self.rho:
            ip = self.wrap_i(i + 1)
            rho_l, vel_l, gr_l, gu_l, rho_r, vel_r, gr_r, gu_r = self.reconstruct_x_face(i, j, ip, j)
            self.face_x[i, j] = self.equilibrium_flux_local(rho_l, vel_l, gr_l, gu_l, rho_r, vel_r, gr_r, gu_r)

    @ti.kernel
    def compute_face_fluxes_y(self):
        for i, j in self.rho:
            jp = self.wrap_j(j + 1)
            gr_l_w, gu_l_w = self.cell_gradients_world(i, j)
            gr_r_w, gu_r_w = self.cell_gradients_world(i, jp)

            rho_l = ti.max(self.rho[i, j] + 0.5 * gr_l_w[1], 1.0e-8)
            rho_r = ti.max(self.rho[i, jp] - 0.5 * gr_r_w[1], 1.0e-8)
            vel_l_w = self.u[i, j] + 0.5 * ti.Vector([gu_l_w[0, 1], gu_l_w[1, 1]])
            vel_r_w = self.u[i, jp] - 0.5 * ti.Vector([gu_r_w[0, 1], gu_r_w[1, 1]])

            vel_l, gr_l, gu_l = self.rotate_y_state(vel_l_w, gr_l_w, gu_l_w)
            vel_r, gr_r, gu_r = self.rotate_y_state(vel_r_w, gr_r_w, gu_r_w)
            flux_l = self.equilibrium_flux_local(rho_l, vel_l, gr_l, gu_l, rho_r, vel_r, gr_r, gu_r)
            self.face_y[i, j] = ti.Vector([flux_l[0], flux_l[2], flux_l[1]])

    def compute_face_fluxes(self):
        self.compute_face_fluxes_x()
        self.compute_face_fluxes_y()
        self.transport_particles()

    @ti.kernel
    def transport_particles(self):
        for k in range(self.max_particles):
            x0 = self.p_x[k]
            y0 = self.p_y[k]
            vx = self.p_vx[k]
            vy = self.p_vy[k]
            mass = self.p_mass[k]
            free_t = ti.min(self.p_tfree[k], ti.static(self.dt))

            i = ti.cast(ti.floor(x0), ti.i32)
            j = ti.cast(ti.floor(y0), ti.i32)
            i = self.wrap_i(i)
            j = self.wrap_j(j)

            sign_x = 0.0
            face_i = i
            cross_tx = ti.static(self.dt) + 1.0
            if vx > 1.0e-12:
                cross_tx = (ti.floor(x0) + 1.0 - x0) / vx
                sign_x = 1.0
                face_i = i
            elif vx < -1.0e-12:
                cross_tx = (x0 - ti.floor(x0)) / (-vx)
                sign_x = -1.0
                face_i = self.wrap_i(i - 1)

            if cross_tx <= free_t:
                scale = sign_x * mass / ti.static(self.dt)
                ti.atomic_add(self.face_x[face_i, j][0], scale)
                ti.atomic_add(self.face_x[face_i, j][1], scale * vx)
                ti.atomic_add(self.face_x[face_i, j][2], scale * vy)

            sign_y = 0.0
            face_j = j
            cross_ty = ti.static(self.dt) + 1.0
            if vy > 1.0e-12:
                cross_ty = (ti.floor(y0) + 1.0 - y0) / vy
                sign_y = 1.0
                face_j = j
            elif vy < -1.0e-12:
                cross_ty = (y0 - ti.floor(y0)) / (-vy)
                sign_y = -1.0
                face_j = self.wrap_j(j - 1)

            if cross_ty <= free_t:
                scale = sign_y * mass / ti.static(self.dt)
                ti.atomic_add(self.face_y[i, face_j][0], scale)
                ti.atomic_add(self.face_y[i, face_j][1], scale * vx)
                ti.atomic_add(self.face_y[i, face_j][2], scale * vy)

            x1 = x0 + vx * free_t
            y1 = y0 + vy * free_t
            if x1 < 0.0:
                x1 += ti.cast(self.nx, ti.f32)
            elif x1 >= ti.cast(self.nx, ti.f32):
                x1 -= ti.cast(self.nx, ti.f32)
            if y1 < 0.0:
                y1 += ti.cast(self.ny, ti.f32)
            elif y1 >= ti.cast(self.ny, ti.f32):
                y1 -= ti.cast(self.ny, ti.f32)
            self.p_x[k] = x1
            self.p_y[k] = y1

    @ti.kernel
    def update(self):
        for i, j in self.rho:
            im = self.wrap_i(i - 1)
            jm = self.wrap_j(j - 1)
            div = (self.face_x[i, j] - self.face_x[im, j]) + (self.face_y[i, j] - self.face_y[i, jm])
            rho_old = self.rho[i, j]
            mom_old = rho_old * self.u[i, j]
            rho_new = ti.max(rho_old - self.dt * div[0], 1.0e-8)
            mom_new = mom_old - self.dt * ti.Vector([div[1], div[2]])
            self.rho_next[i, j] = rho_new
            self.u_next[i, j] = mom_new / rho_new

    @ti.kernel
    def swap(self):
        for i, j in self.rho:
            self.rho[i, j] = self.rho_next[i, j]
            self.u[i, j] = self.u_next[i, j]

    @ti.kernel
    def resample_particles(self):
        for i, j in self.rho:
            rho = self.rho[i, j]
            vel = self.u[i, j]
            for p in range(self.particles_per_cell):
                k = self.particle_index(i, j, p)
                z0, z1 = self.normal_pair()
                self.p_x[k] = ti.cast(i, ti.f32) + ti.random(ti.f32)
                self.p_y[k] = ti.cast(j, ti.f32) + ti.random(ti.f32)
                self.p_vx[k] = vel[0] + ti.static(self.cs) * z0
                self.p_vy[k] = vel[1] + ti.static(self.cs) * z1
                self.p_mass[k] = rho / ti.cast(self.particles_per_cell, ti.f32)
                self.p_tfree[k] = -ti.static(self.tau) * ti.log(ti.max(ti.random(ti.f32), 1.0e-6))

    def step(self):
        self.compute_face_fluxes()
        self.update()
        self.swap()
        self.resample_particles()

    def kinetic_energy(self):
        rho_np = self.rho.to_numpy()
        u_np = self.u.to_numpy()
        return float(0.5 * np.mean(rho_np * np.sum(u_np * u_np, axis=-1)))

    def density_stats(self):
        rho_np = self.rho.to_numpy()
        return float(rho_np.min()), float(rho_np.mean()), float(rho_np.max())

    def snapshot_arrays(self):
        return self.u.to_numpy()[:, :, 0]

    def shear_wave_error(self, time):
        ux = self.snapshot_arrays()
        y = (np.arange(self.ny, dtype=np.float32) + 0.5) / float(self.ny)
        amp0 = float(self.mach) * self.cs
        k_eff2 = 4.0 * math.sin(math.pi / float(self.ny)) ** 2
        amp = amp0 * math.exp(-self.nu * k_eff2 * time)
        exact_1d = amp * np.sin(2.0 * math.pi * y)
        exact_ux = np.broadcast_to(exact_1d[None, :], ux.shape)
        err_ux = ux - exact_ux
        l2 = float(np.sqrt(np.mean(err_ux * err_ux)))
        linf = float(np.max(np.abs(err_ux)))
        amp_num = float(2.0 * np.mean(ux * np.sin(2.0 * math.pi * y[None, :])))
        return l2, linf, amp_num, amp

    def render_image(self):
        values = self.snapshot_arrays()
        vmax = max(1.0e-12, float(np.max(np.abs(values))))
        normalized = np.clip(0.5 + 0.5 * values / vmax, 0.0, 1.0)
        image = np.zeros((self.nx, self.ny, 3), dtype=np.float32)
        image[:, :, 0] = normalized
        image[:, :, 1] = 0.2 + 0.6 * (1.0 - np.abs(2.0 * normalized - 1.0))
        image[:, :, 2] = 1.0 - normalized
        return image


SinglePhaseGKS2D = SinglePhaseUGKWP2D


def main():
    ti.init(arch=ti.gpu)
    sim = SinglePhaseUGKWP2D()
    sim.init()
    gui = ti.GUI("BGK-UGKWP", res=(sim.nx, sim.ny), fast_gui=False)
    s = 0
    sim_time = 0.0
    while gui.running:
        # print(s, sim_time)
        sim.step()
        gui.set_image(sim.render_image())
        gui.show()
        s += 1
        sim_time += sim.dt
        if s % 100 == 0:
            l2, linf, amp_num, amp_exact = sim.shear_wave_error(sim_time)
            print(f" l2={l2:.6e} linf={linf:.6e} amp_num={amp_num:.6e} amp_exact={amp_exact:.6e}")


if __name__ == "__main__":
    main()
