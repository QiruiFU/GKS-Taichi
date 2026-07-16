import math
from pathlib import Path

import numpy as np
import taichi as ti


@ti.data_oriented
class SinglePhaseGKS3D:
    """Isothermal 3D BGK-GKS for a periodic Taylor-Green vortex.

    The conserved variables are [rho, rho*u, rho*v, rho*w].  Temperature is
    fixed through cs^2, so this is the weakly-compressible/isothermal form of
    GKS rather than a total-energy formulation.
    """

    def __init__(self, nx=128, ny=128, nz=128, reynolds=16000.0):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dt = 0.25
        self.cs = 1.0 / math.sqrt(3.0)
        self.cs2 = self.cs * self.cs
        self.mach = 0.1
        self.reynolds = reynolds
        # Relative to the Taylor-Green velocity scale.  This deterministic
        # solenoidal perturbation breaks the exact discrete symmetries and
        # seeds the three-dimensional instability reproducibly.
        self.perturbation_amplitude = 0.0

        # Grid spacing is one in the solver.  For sin(2 pi x / nx), the
        # standard Taylor-Green length scale 1/k is nx/(2 pi).
        vortex_length = min(self.nx, self.ny, self.nz) / (2.0 * math.pi)
        self.nu = (self.mach * self.cs) * vortex_length / self.reynolds
        self.tau = self.nu / self.cs2

        shape = (self.nx, self.ny, self.nz)
        self.rho = ti.field(ti.f32, shape=shape)
        self.rho_next = ti.field(ti.f32, shape=shape)
        self.u = ti.Vector.field(3, ti.f32, shape=shape)
        self.u_next = ti.Vector.field(3, ti.f32, shape=shape)
        self.face_x = ti.Vector.field(4, ti.f32, shape=shape)
        self.face_y = ti.Vector.field(4, ti.f32, shape=shape)
        self.face_z = ti.Vector.field(4, ti.f32, shape=shape)

        # A two-dimensional field avoids copying the whole 3D velocity field
        # back to Python on every rendered frame.
        self.slice_vorticity_magnitude = ti.field(ti.f32, shape=(self.nx, self.ny))

    @ti.func
    def wrap_i(self, i):
        return (i + self.nx) % self.nx

    @ti.func
    def wrap_j(self, j):
        return (j + self.ny) % self.ny

    @ti.func
    def wrap_k(self, k):
        return (k + self.nz) % self.nz

    @ti.func
    def central_slope(self, qm, qp):
        """Unlimited second-order centered slope on the unit-spaced grid."""
        return 0.5 * (qp - qm)

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
    def full_moments_1d(self, velocity):
        cs2 = ti.static(self.cs2)
        cs4 = ti.static(self.cs2 * self.cs2)
        velocity2 = velocity * velocity
        moments = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        moments[0] = 1.0
        moments[1] = velocity
        moments[2] = velocity2 + cs2
        moments[3] = velocity * velocity2 + 3.0 * cs2 * velocity
        moments[4] = velocity2 * velocity2 + 6.0 * cs2 * velocity2 + 3.0 * cs4
        return moments

    @ti.func
    def half_moments_normal(self, velocity):
        """int_{xi>0} xi^k N(velocity, cs^2) dxi for k=0,...,4."""
        cs = ti.static(self.cs)
        cs2 = ti.static(self.cs2)
        inv_sqrt2 = ti.static(1.0 / math.sqrt(2.0))
        inv_sqrt2pi = ti.static(1.0 / math.sqrt(2.0 * math.pi))
        z = velocity / cs
        phi = ti.exp(-0.5 * z * z) * inv_sqrt2pi
        moments = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
        moments[0] = 0.5 * (1.0 + self.erf_approx(z * inv_sqrt2))
        moments[1] = velocity * moments[0] + cs * phi
        for n in ti.static(range(2, 5)):
            moments[n] = velocity * moments[n - 1] + (n - 1) * cs2 * moments[n - 2]
        return moments

    @ti.func
    def moment3(self, mx, my, mz, ix, iy, iz):
        return mx[ix] * my[iy] * mz[iz]

    @ti.func
    def component_powers(self, component: ti.template()):
        rx, ry, rz = 0, 0, 0
        if ti.static(component == 1):
            rx = 1
        elif ti.static(component == 2):
            ry = 1
        elif ti.static(component == 3):
            rz = 1
        return rx, ry, rz

    @ti.func
    def state_moment(self, mx, my, mz, density, component: ti.template(), direction: ti.template()):
        rx, ry, rz = self.component_powers(component)
        if ti.static(direction == 0):
            rx += 1
        elif ti.static(direction == 1):
            ry += 1
        else:
            rz += 1
        return density * self.moment3(mx, my, mz, rx, ry, rz)

    @ti.func
    def coefficient_moment(self, mx, my, mz, coeff, component: ti.template(), direction0: ti.template(), direction1: ti.template()):
        """Moment of psi * xi_direction0 * xi_direction1 * coeff*M.

        direction1 can be -1, in which case only one velocity factor is used.
        """
        rx, ry, rz = self.component_powers(component)
        if ti.static(direction0 == 0):
            rx += 1
        elif ti.static(direction0 == 1):
            ry += 1
        else:
            rz += 1
        if ti.static(direction1 == 0):
            rx += 1
        elif ti.static(direction1 == 1):
            ry += 1
        elif ti.static(direction1 == 2):
            rz += 1
        return (
            coeff[0] * self.moment3(mx, my, mz, rx, ry, rz)
            + coeff[1] * self.moment3(mx, my, mz, rx + 1, ry, rz)
            + coeff[2] * self.moment3(mx, my, mz, rx, ry + 1, rz)
            + coeff[3] * self.moment3(mx, my, mz, rx, ry, rz + 1)
        )

    @ti.func
    def solve_coefficients(self, h, velocity):
        cs2_inv = ti.static(1.0 / self.cs2)
        coeff = ti.Vector([0.0, 0.0, 0.0, 0.0])
        coeff[1] = cs2_inv * (h[1] - velocity[0] * h[0])
        coeff[2] = cs2_inv * (h[2] - velocity[1] * h[0])
        coeff[3] = cs2_inv * (h[3] - velocity[2] * h[0])
        coeff[0] = h[0] - velocity[0] * coeff[1] - velocity[1] * coeff[2] - velocity[2] * coeff[3]
        return coeff

    @ti.func
    def spatial_coefficients(self, density, velocity, grad_density, grad_velocity):
        cs2_inv = ti.static(1.0 / self.cs2)
        a = ti.Vector([0.0, 0.0, 0.0, 0.0])
        b = ti.Vector([0.0, 0.0, 0.0, 0.0])
        c = ti.Vector([0.0, 0.0, 0.0, 0.0])
        for direction in ti.static(range(3)):
            coeff = ti.Vector([0.0, 0.0, 0.0, 0.0])
            velocity_dot_gradient = 0.0
            for component in ti.static(range(3)):
                velocity_dot_gradient += velocity[component] * grad_velocity[component, direction]
                coeff[component + 1] = density * grad_velocity[component, direction] * cs2_inv
            coeff[0] = grad_density[direction] - density * velocity_dot_gradient * cs2_inv
            if direction == 0:
                a = coeff
            elif direction == 1:
                b = coeff
            else:
                c = coeff
        return a, b, c

    @ti.func
    def temporal_coefficients(self, velocity, mx, my, mz, a, b, c):
        h = ti.Vector([0.0, 0.0, 0.0, 0.0])
        for component in ti.static(range(4)):
            h[component] = -(
                self.coefficient_moment(mx, my, mz, a, component, 0, -1)
                + self.coefficient_moment(mx, my, mz, b, component, 1, -1)
                + self.coefficient_moment(mx, my, mz, c, component, 2, -1)
            )
        return self.solve_coefficients(h, velocity)

    @ti.func
    def flux_moment(self, mx, my, mz, density, a, b, c, A, component: ti.template(), kind: ti.template()):
        # This is the flux through a face whose local normal is x.
        out = 0.0
        if ti.static(kind == 0):
            out = self.state_moment(mx, my, mz, density, component, 0)
        elif ti.static(kind == 1):
            out = (
                self.coefficient_moment(mx, my, mz, a, component, 0, 0)
                + self.coefficient_moment(mx, my, mz, b, component, 1, 0)
                + self.coefficient_moment(mx, my, mz, c, component, 2, 0)
            )
        else:
            out = self.coefficient_moment(mx, my, mz, A, component, 0, -1)
        return out

    @ti.func
    def interface_equilibrium(self, density_l, mx_l, my_l, mz_l, hp_l, density_r, mx_r, my_r, mz_r, hp_r):
        hn_r = mx_r - hp_r
        w = ti.Vector([0.0, 0.0, 0.0, 0.0])
        w[0] = density_l * hp_l[0] * my_l[0] * mz_l[0] + density_r * hn_r[0] * my_r[0] * mz_r[0]
        w[1] = density_l * hp_l[1] * my_l[0] * mz_l[0] + density_r * hn_r[1] * my_r[0] * mz_r[0]
        w[2] = density_l * hp_l[0] * my_l[1] * mz_l[0] + density_r * hn_r[0] * my_r[1] * mz_r[0]
        w[3] = density_l * hp_l[0] * my_l[0] * mz_l[1] + density_r * hn_r[0] * my_r[0] * mz_r[1]
        density0 = ti.max(w[0], 1.0e-8)
        velocity0 = ti.Vector([w[1] / density0, w[2] / density0, w[3] / density0])
        return density0, velocity0

    @ti.func
    def gks_flux_local(self, density_l, velocity_l, grad_density_l, grad_velocity_l, density_r, velocity_r, grad_density_r, grad_velocity_r):
        mx_l = self.full_moments_1d(velocity_l[0])
        my_l = self.full_moments_1d(velocity_l[1])
        mz_l = self.full_moments_1d(velocity_l[2])
        mx_r = self.full_moments_1d(velocity_r[0])
        my_r = self.full_moments_1d(velocity_r[1])
        mz_r = self.full_moments_1d(velocity_r[2])
        hp_l = self.half_moments_normal(velocity_l[0])
        hp_r = self.half_moments_normal(velocity_r[0])
        hn_r = mx_r - hp_r

        a_l, b_l, c_l = self.spatial_coefficients(density_l, velocity_l, grad_density_l, grad_velocity_l)
        a_r, b_r, c_r = self.spatial_coefficients(density_r, velocity_r, grad_density_r, grad_velocity_r)
        A_l = self.temporal_coefficients(velocity_l, mx_l, my_l, mz_l, a_l, b_l, c_l)
        A_r = self.temporal_coefficients(velocity_r, mx_r, my_r, mz_r, a_r, b_r, c_r)

        density0, velocity0 = self.interface_equilibrium(density_l, mx_l, my_l, mz_l, hp_l, density_r, mx_r, my_r, mz_r, hp_r)
        mx0 = self.full_moments_1d(velocity0[0])
        my0 = self.full_moments_1d(velocity0[1])
        mz0 = self.full_moments_1d(velocity0[2])
        grad_density0 = ti.Vector([
            0.5 * (grad_density_l[0] + grad_density_r[0]),
            0.5 * (grad_density_l[1] + grad_density_r[1]),
            0.5 * (grad_density_l[2] + grad_density_r[2]),
        ])
        grad_velocity0 = ti.Matrix.zero(ti.f32, 3, 3)
        for row in ti.static(range(3)):
            for column in ti.static(range(3)):
                grad_velocity0[row, column] = 0.5 * (grad_velocity_l[row, column] + grad_velocity_r[row, column])
        a0, b0, c0 = self.spatial_coefficients(density0, velocity0, grad_density0, grad_velocity0)
        A0 = self.temporal_coefficients(velocity0, mx0, my0, mz0, a0, b0, c0)

        tau = ti.static(self.tau)
        dt = ti.static(self.dt)
        E = ti.exp(-dt / tau)
        c_eq = 1.0 - tau * (1.0 - E) / dt
        c_eq_space = (2.0 * tau * tau * (1.0 - E) - tau * dt * E - tau * dt) / dt
        c_eq_time = 0.5 * dt - tau + tau * tau * (1.0 - E) / dt
        c_init = tau * (1.0 - E) / dt
        c_init_space = -(2.0 * tau * tau * (1.0 - E) - tau * dt * E) / dt
        c_init_time = -tau * tau * (1.0 - E) / dt

        flux = ti.Vector([0.0, 0.0, 0.0, 0.0])
        for component in ti.static(range(4)):
            eq = self.flux_moment(mx0, my0, mz0, density0, a0, b0, c0, A0, component, 0)
            eq_space = self.flux_moment(mx0, my0, mz0, density0, a0, b0, c0, A0, component, 1)
            eq_time = self.flux_moment(mx0, my0, mz0, density0, a0, b0, c0, A0, component, 2)
            init = (
                self.flux_moment(hp_l, my_l, mz_l, density_l, a_l, b_l, c_l, A_l, component, 0)
                + self.flux_moment(hn_r, my_r, mz_r, density_r, a_r, b_r, c_r, A_r, component, 0)
            )
            init_space = (
                self.flux_moment(hp_l, my_l, mz_l, density_l, a_l, b_l, c_l, A_l, component, 1)
                + self.flux_moment(hn_r, my_r, mz_r, density_r, a_r, b_r, c_r, A_r, component, 1)
            )
            init_time = (
                self.flux_moment(hp_l, my_l, mz_l, density_l, a_l, b_l, c_l, A_l, component, 2)
                + self.flux_moment(hn_r, my_r, mz_r, density_r, a_r, b_r, c_r, A_r, component, 2)
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
    def cell_gradients_world(self, i, j, k):
        ip, im = self.wrap_i(i + 1), self.wrap_i(i - 1)
        jp, jm = self.wrap_j(j + 1), self.wrap_j(j - 1)
        kp, km = self.wrap_k(k + 1), self.wrap_k(k - 1)
        grad_density = ti.Vector([
            self.central_slope(self.rho[im, j, k], self.rho[ip, j, k]),
            self.central_slope(self.rho[i, jm, k], self.rho[i, jp, k]),
            self.central_slope(self.rho[i, j, km], self.rho[i, j, kp]),
        ])
        grad_velocity = ti.Matrix.zero(ti.f32, 3, 3)
        for component in ti.static(range(3)):
            grad_velocity[component, 0] = self.central_slope(self.u[im, j, k][component], self.u[ip, j, k][component])
            grad_velocity[component, 1] = self.central_slope(self.u[i, jm, k][component], self.u[i, jp, k][component])
            grad_velocity[component, 2] = self.central_slope(self.u[i, j, km][component], self.u[i, j, kp][component])
        return grad_density, grad_velocity

    @ti.func
    def reconstruct_x_face(self, il, j, k, ir):
        grad_density_l, grad_velocity_l = self.cell_gradients_world(il, j, k)
        grad_density_r, grad_velocity_r = self.cell_gradients_world(ir, j, k)
        density_l = ti.max(self.rho[il, j, k] + 0.5 * grad_density_l[0], 1.0e-8)
        density_r = ti.max(self.rho[ir, j, k] - 0.5 * grad_density_r[0], 1.0e-8)
        velocity_l = ti.Vector([
            self.u[il, j, k][0] + 0.5 * grad_velocity_l[0, 0],
            self.u[il, j, k][1] + 0.5 * grad_velocity_l[1, 0],
            self.u[il, j, k][2] + 0.5 * grad_velocity_l[2, 0],
        ])
        velocity_r = ti.Vector([
            self.u[ir, j, k][0] - 0.5 * grad_velocity_r[0, 0],
            self.u[ir, j, k][1] - 0.5 * grad_velocity_r[1, 0],
            self.u[ir, j, k][2] - 0.5 * grad_velocity_r[2, 0],
        ])
        return density_l, velocity_l, grad_density_l, grad_velocity_l, density_r, velocity_r, grad_density_r, grad_velocity_r

    @ti.func
    def rotate_y_state(self, velocity_world, grad_density_world, grad_velocity_world):
        """World (x, y, z) -> local (normal=y, tangent=x, tangent=z)."""
        velocity_local = ti.Vector([velocity_world[1], velocity_world[0], velocity_world[2]])
        grad_density_local = ti.Vector([grad_density_world[1], grad_density_world[0], grad_density_world[2]])
        grad_velocity_local = ti.Matrix.zero(ti.f32, 3, 3)
        grad_velocity_local[0, 0] = grad_velocity_world[1, 1]
        grad_velocity_local[0, 1] = grad_velocity_world[1, 0]
        grad_velocity_local[0, 2] = grad_velocity_world[1, 2]
        grad_velocity_local[1, 0] = grad_velocity_world[0, 1]
        grad_velocity_local[1, 1] = grad_velocity_world[0, 0]
        grad_velocity_local[1, 2] = grad_velocity_world[0, 2]
        grad_velocity_local[2, 0] = grad_velocity_world[2, 1]
        grad_velocity_local[2, 1] = grad_velocity_world[2, 0]
        grad_velocity_local[2, 2] = grad_velocity_world[2, 2]
        return velocity_local, grad_density_local, grad_velocity_local

    @ti.func
    def rotate_z_state(self, velocity_world, grad_density_world, grad_velocity_world):
        """World (x, y, z) -> local (normal=z, tangent=x, tangent=y)."""
        velocity_local = ti.Vector([velocity_world[2], velocity_world[0], velocity_world[1]])
        grad_density_local = ti.Vector([grad_density_world[2], grad_density_world[0], grad_density_world[1]])
        grad_velocity_local = ti.Matrix.zero(ti.f32, 3, 3)
        grad_velocity_local[0, 0] = grad_velocity_world[2, 2]
        grad_velocity_local[0, 1] = grad_velocity_world[2, 0]
        grad_velocity_local[0, 2] = grad_velocity_world[2, 1]
        grad_velocity_local[1, 0] = grad_velocity_world[0, 2]
        grad_velocity_local[1, 1] = grad_velocity_world[0, 0]
        grad_velocity_local[1, 2] = grad_velocity_world[0, 1]
        grad_velocity_local[2, 0] = grad_velocity_world[1, 2]
        grad_velocity_local[2, 1] = grad_velocity_world[1, 0]
        grad_velocity_local[2, 2] = grad_velocity_world[1, 1]
        return velocity_local, grad_density_local, grad_velocity_local

    @ti.kernel
    def init(self):
        for i, j, k in self.rho:
            x = 2.0 * math.pi * (ti.cast(i, ti.f32) + 0.5) / ti.cast(self.nx, ti.f32)
            y = 2.0 * math.pi * (ti.cast(j, ti.f32) + 0.5) / ti.cast(self.ny, ti.f32)
            z = 2.0 * math.pi * (ti.cast(k, ti.f32) + 0.5) / ti.cast(self.nz, ti.f32)
            v0 = self.mach * ti.static(self.cs)
            eps = self.perturbation_amplitude * v0
            density = 1.0 + v0 * v0 / (16.0 * ti.static(self.cs2)) * (ti.cos(2.0 * x) + ti.cos(2.0 * y)) * (ti.cos(2.0 * z) + 2.0)
            # Each perturbation component is independent of its own spatial
            # coordinate, so du'/dx + dv'/dy + dw'/dz = 0 exactly.
            ux = v0 * ti.sin(x) * ti.cos(y) * ti.cos(z) + eps * ti.sin(2.0 * y + 0.37) * ti.sin(3.0 * z + 0.19)
            uy = -v0 * ti.cos(x) * ti.sin(y) * ti.cos(z) + eps * ti.sin(3.0 * z + 0.41) * ti.sin(2.0 * x + 0.23)
            uz = eps * ti.sin(2.0 * x + 0.11) * ti.sin(3.0 * y + 0.29)
            self.rho[i, j, k] = density
            self.u[i, j, k] = ti.Vector([ux, uy, uz])

    @ti.kernel
    def compute_face_fluxes_x(self):
        for i, j, k in self.rho:
            ip = self.wrap_i(i + 1)
            density_l, velocity_l, grad_density_l, grad_velocity_l, density_r, velocity_r, grad_density_r, grad_velocity_r = self.reconstruct_x_face(i, j, k, ip)
            self.face_x[i, j, k] = self.gks_flux_local(density_l, velocity_l, grad_density_l, grad_velocity_l, density_r, velocity_r, grad_density_r, grad_velocity_r)

    @ti.kernel
    def compute_face_fluxes_y(self):
        for i, j, k in self.rho:
            jp = self.wrap_j(j + 1)
            grad_density_l_w, grad_velocity_l_w = self.cell_gradients_world(i, j, k)
            grad_density_r_w, grad_velocity_r_w = self.cell_gradients_world(i, jp, k)
            density_l = ti.max(self.rho[i, j, k] + 0.5 * grad_density_l_w[1], 1.0e-8)
            density_r = ti.max(self.rho[i, jp, k] - 0.5 * grad_density_r_w[1], 1.0e-8)
            velocity_l_w = ti.Vector([
                self.u[i, j, k][0] + 0.5 * grad_velocity_l_w[0, 1],
                self.u[i, j, k][1] + 0.5 * grad_velocity_l_w[1, 1],
                self.u[i, j, k][2] + 0.5 * grad_velocity_l_w[2, 1],
            ])
            velocity_r_w = ti.Vector([
                self.u[i, jp, k][0] - 0.5 * grad_velocity_r_w[0, 1],
                self.u[i, jp, k][1] - 0.5 * grad_velocity_r_w[1, 1],
                self.u[i, jp, k][2] - 0.5 * grad_velocity_r_w[2, 1],
            ])
            velocity_l, grad_density_l, grad_velocity_l = self.rotate_y_state(velocity_l_w, grad_density_l_w, grad_velocity_l_w)
            velocity_r, grad_density_r, grad_velocity_r = self.rotate_y_state(velocity_r_w, grad_density_r_w, grad_velocity_r_w)
            local_flux = self.gks_flux_local(density_l, velocity_l, grad_density_l, grad_velocity_l, density_r, velocity_r, grad_density_r, grad_velocity_r)
            self.face_y[i, j, k] = ti.Vector([local_flux[0], local_flux[2], local_flux[1], local_flux[3]])

    @ti.kernel
    def compute_face_fluxes_z(self):
        for i, j, k in self.rho:
            kp = self.wrap_k(k + 1)
            grad_density_l_w, grad_velocity_l_w = self.cell_gradients_world(i, j, k)
            grad_density_r_w, grad_velocity_r_w = self.cell_gradients_world(i, j, kp)
            density_l = ti.max(self.rho[i, j, k] + 0.5 * grad_density_l_w[2], 1.0e-8)
            density_r = ti.max(self.rho[i, j, kp] - 0.5 * grad_density_r_w[2], 1.0e-8)
            velocity_l_w = ti.Vector([
                self.u[i, j, k][0] + 0.5 * grad_velocity_l_w[0, 2],
                self.u[i, j, k][1] + 0.5 * grad_velocity_l_w[1, 2],
                self.u[i, j, k][2] + 0.5 * grad_velocity_l_w[2, 2],
            ])
            velocity_r_w = ti.Vector([
                self.u[i, j, kp][0] - 0.5 * grad_velocity_r_w[0, 2],
                self.u[i, j, kp][1] - 0.5 * grad_velocity_r_w[1, 2],
                self.u[i, j, kp][2] - 0.5 * grad_velocity_r_w[2, 2],
            ])
            velocity_l, grad_density_l, grad_velocity_l = self.rotate_z_state(velocity_l_w, grad_density_l_w, grad_velocity_l_w)
            velocity_r, grad_density_r, grad_velocity_r = self.rotate_z_state(velocity_r_w, grad_density_r_w, grad_velocity_r_w)
            local_flux = self.gks_flux_local(density_l, velocity_l, grad_density_l, grad_velocity_l, density_r, velocity_r, grad_density_r, grad_velocity_r)
            self.face_z[i, j, k] = ti.Vector([local_flux[0], local_flux[2], local_flux[3], local_flux[1]])

    def compute_face_fluxes(self):
        self.compute_face_fluxes_x()
        self.compute_face_fluxes_y()
        self.compute_face_fluxes_z()

    @ti.kernel
    def update(self):
        for i, j, k in self.rho:
            im, jm, km = self.wrap_i(i - 1), self.wrap_j(j - 1), self.wrap_k(k - 1)
            div = (
                self.face_x[i, j, k] - self.face_x[im, j, k]
                + self.face_y[i, j, k] - self.face_y[i, jm, k]
                + self.face_z[i, j, k] - self.face_z[i, j, km]
            )
            density_old = self.rho[i, j, k]
            momentum_old = ti.Vector([
                density_old * self.u[i, j, k][0],
                density_old * self.u[i, j, k][1],
                density_old * self.u[i, j, k][2],
            ])
            density_new = ti.max(density_old - self.dt * div[0], 1.0e-8)
            momentum_new = ti.Vector([
                momentum_old[0] - self.dt * div[1],
                momentum_old[1] - self.dt * div[2],
                momentum_old[2] - self.dt * div[3],
            ])
            self.rho_next[i, j, k] = density_new
            self.u_next[i, j, k] = ti.Vector([
                momentum_new[0] / density_new,
                momentum_new[1] / density_new,
                momentum_new[2] / density_new,
            ])

    @ti.kernel
    def swap(self):
        for i, j, k in self.rho:
            self.rho[i, j, k] = self.rho_next[i, j, k]
            self.u[i, j, k] = self.u_next[i, j, k]

    @ti.kernel
    def update_render_slice(self):
        slice_k = ti.static(self.nz // 2)
        for i, j in self.slice_vorticity_magnitude:
            ip, im = self.wrap_i(i + 1), self.wrap_i(i - 1)
            jp, jm = self.wrap_j(j + 1), self.wrap_j(j - 1)
            kp, km = self.wrap_k(slice_k + 1), self.wrap_k(slice_k - 1)
            omega_x = 0.5 * (
                self.u[i, jp, slice_k][2] - self.u[i, jm, slice_k][2]
                - self.u[i, j, kp][1] + self.u[i, j, km][1]
            )
            omega_y = 0.5 * (
                self.u[i, j, kp][0] - self.u[i, j, km][0]
                - self.u[ip, j, slice_k][2] + self.u[im, j, slice_k][2]
            )
            omega_z = 0.5 * (
                self.u[ip, j, slice_k][1] - self.u[im, j, slice_k][1]
                - self.u[i, jp, slice_k][0] + self.u[i, jm, slice_k][0]
            )
            self.slice_vorticity_magnitude[i, j] = ti.sqrt(omega_x * omega_x + omega_y * omega_y + omega_z * omega_z)

    def step(self):
        self.compute_face_fluxes()
        self.update()
        self.swap()

    def kinetic_energy(self):
        density = self.rho.to_numpy()
        velocity = self.u.to_numpy()
        return float(0.5 * np.mean(density * np.sum(velocity * velocity, axis=-1)))

    def density_stats(self):
        density = self.rho.to_numpy()
        return float(density.min()), float(density.mean()), float(density.max())

    def render_image(self):
        self.update_render_slice()
        values = self.slice_vorticity_magnitude.to_numpy()
        vmax = max(1.0e-12, float(np.max(np.abs(values))))
        normalized = np.clip(values / vmax, 0.0, 1.0)
        image = np.zeros((self.nx, self.ny, 3), dtype=np.float32)
        image[:, :, 0] = normalized
        image[:, :, 1] = 0.2 + 0.6 * (1.0 - np.abs(2.0 * normalized - 1.0))
        image[:, :, 2] = 1.0 - normalized
        return image


def main():
    ti.init(arch=ti.gpu)
    sim = SinglePhaseGKS3D()
    sim.init()
    gui = ti.GUI("3D BGK-GKS: Taylor-Green vortex, z-midplane |vorticity|", res=(sim.nx, sim.ny), fast_gui=False)
    frame_dir = Path("tgv_frames")
    frame_dir.mkdir(exist_ok=True)
    save_interval = 500
    step = 0
    simulation_time = 0.0
    while gui.running:
        sim.step()
        step += 1
        simulation_time += sim.dt
        if step % save_interval == 0:
            gui.set_image(sim.render_image())
            frame_path = frame_dir / f"tgv_zmid_omega_mag_{step:08d}.png"
            gui.show(str(frame_path))
            print(f"saved {frame_path}")
        # if step % 100 == 0:
        #     density_min, density_mean, density_max = sim.density_stats()
        #     print(
        #         f"step={step} time={simulation_time:.3f} KE={sim.kinetic_energy():.6e} "
        #         f"rho=({density_min:.6e}, {density_mean:.6e}, {density_max:.6e})"
        #     )


if __name__ == "__main__":
    main()
