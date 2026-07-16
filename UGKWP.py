import math
import argparse
import os
from pathlib import Path

import numpy as np
import taichi as ti


@ti.data_oriented
class SinglePhaseUGKWP2D:
    """2D UGKWP-style solver with mass, momentum, and energy.

    Conserved variables are [rho, rho*u, rho*v, rho*E].  The flux solver uses
    the UGKWP wave-particle split from the integral BGK solution.  The
    collisional part is evolved by the analytic gas-kinetic wave flux, while
    the no-collision probability exp(-dt/tau) part is represented by
    simulation particles and contributes a finite-volume crossing flux.

    The benchmark-facing rho/u initialization, diagnostics, and main loop are
    kept compatible with GKS.py.  This file intentionally changes the step
    procedure and its helper kernels rather than the surrounding framework.
    """

    def __init__(self, nx=256, ny=256, particles_per_cell=300, kn=0.1):
        self.nx = nx
        self.ny = ny
        self.particles_per_cell = particles_per_cell
        self.max_particles = self.nx * self.ny * self.particles_per_cell
        self.dt = 0.005
        self.cs = 1.0 / math.sqrt(3.0)
        self.cs2 = self.cs * self.cs
        self.nu = 0.00001
        self.mach = 0.3
        self.gamma = 1.4
        self.kn = kn
        self.temperature_ref = 1.0
        self.viscosity_index = 0.81
        self.mu_ref = 15.0 * math.sqrt(math.pi) * self.kn / (
            2.0 * (5.0 - 2.0 * self.viscosity_index) * (7.0 - 2.0 * self.viscosity_index)
        )

        self.rho = ti.field(ti.f64, shape=(self.nx, self.ny))
        self.rho_next = ti.field(ti.f64, shape=(self.nx, self.ny))
        self.rhoE = ti.field(ti.f64, shape=(self.nx, self.ny))
        self.rhoE_next = ti.field(ti.f64, shape=(self.nx, self.ny))
        self.u = ti.Vector.field(2, ti.f64, shape=(self.nx, self.ny))
        self.u_next = ti.Vector.field(2, ti.f64, shape=(self.nx, self.ny))
        self.wh = ti.Vector.field(4, ti.f64, shape=(self.nx, self.ny))
        self.wp = ti.Vector.field(4, ti.f64, shape=(self.nx, self.ny))
        self.mu = ti.field(ti.f64, shape=(self.nx, self.ny))
        self.tau = ti.field(ti.f64, shape=(self.nx, self.ny))
        # face_x[i, j] is the face on the right of cell (i, j).  The left
        # physical boundary needs its own storage when x is not periodic.
        self.face_x = ti.Vector.field(4, ti.f64, shape=(self.nx, self.ny))
        self.face_x_left = ti.Vector.field(4, ti.f64, shape=self.ny)
        self.face_y = ti.Vector.field(4, ti.f64, shape=(self.nx, self.ny))
        self.fixed_x_boundaries = ti.field(ti.i32, shape=())
        self.p_x = ti.field(ti.f64, shape=self.max_particles)
        self.p_y = ti.field(ti.f64, shape=self.max_particles)
        self.p_vx = ti.field(ti.f64, shape=self.max_particles)
        self.p_vy = ti.field(ti.f64, shape=self.max_particles)
        self.p_internal = ti.field(ti.f64, shape=self.max_particles)
        self.p_mass = ti.field(ti.f64, shape=self.max_particles)
        self.p_tfree = ti.field(ti.f64, shape=self.max_particles)
        self.p_alive = ti.field(ti.i32, shape=self.max_particles)

    @ti.func
    def wrap_i(self, i):
        return (i + self.nx) % self.nx

    @ti.func
    def wrap_j(self, j):
        return (j + self.ny) % self.ny

    @ti.func
    def particle_index(self, i, j, p):
        return (i * self.ny + j) * self.particles_per_cell + p

    @ti.func
    def normal_pair(self):
        r1 = ti.max(ti.random(ti.f64), 1.0e-12)
        r2 = ti.random(ti.f64)
        mag = ti.sqrt(-2.0 * ti.log(r1))
        ang = 2.0 * math.pi * r2
        return mag * ti.cos(ang), mag * ti.sin(ang)

    @ti.func
    def conservative_from_primitive(self, rho, vel, pressure):
        kinetic = 0.5 * rho * (vel[0] * vel[0] + vel[1] * vel[1])
        rhoE = pressure / (ti.static(self.gamma) - 1.0) + kinetic
        return ti.Vector([rho, rho * vel[0], rho * vel[1], rhoE])

    @ti.func
    def primitive_from_conservative(self, w):
        rho = ti.max(w[0], 1.0e-8)
        vel = ti.Vector([w[1] / rho, w[2] / rho])
        kinetic = 0.5 * rho * (vel[0] * vel[0] + vel[1] * vel[1])
        pressure = ti.max((ti.static(self.gamma) - 1.0) * (w[3] - kinetic), 1.0e-8)
        return rho, vel, pressure

    @ti.func
    def relaxation_from_w(self, w):
        rho, _, pressure = self.primitive_from_conservative(w)
        temperature = pressure / rho
        mu = ti.static(self.mu_ref) * ti.pow(temperature / ti.static(self.temperature_ref), ti.static(self.viscosity_index))
        tau = mu / pressure
        return mu, tau

    @ti.func
    def update_cell_relaxation(self, i, j, w):
        mu, tau = self.relaxation_from_w(w)
        self.mu[i, j] = mu
        self.tau[i, j] = tau

    @ti.func
    def euler_flux_x_from_w(self, w):
        rho, vel, pressure = self.primitive_from_conservative(w)
        return ti.Vector(
            [
                w[1],
                w[1] * vel[0] + pressure,
                w[2] * vel[0],
                (w[3] + pressure) * vel[0],
            ]
        )

    @ti.func
    def euler_flux_y_from_w(self, w):
        rho, vel, pressure = self.primitive_from_conservative(w)
        return ti.Vector(
            [
                w[2],
                w[1] * vel[1],
                w[2] * vel[1] + pressure,
                (w[3] + pressure) * vel[1],
            ]
        )

    @ti.func
    def particle_energy(self, k):
        return self.p_mass[k] * (0.5 * (self.p_vx[k] * self.p_vx[k] + self.p_vy[k] * self.p_vy[k]) + self.p_internal[k])

    @ti.func
    def sample_particle_in_cell(self, k, i, j, rho, vel, pressure):
        z0, z1 = self.normal_pair()
        theta = pressure / ti.max(rho, 1.0e-8)
        thermal = ti.sqrt(ti.max(theta, 1.0e-8))
        internal_dof = 2.0 / (ti.static(self.gamma) - 1.0) - 2.0
        self.p_x[k] = ti.cast(i, ti.f64) + ti.random(ti.f64)
        self.p_y[k] = ti.cast(j, ti.f64) + ti.random(ti.f64)
        self.p_vx[k] = vel[0] + thermal * z0
        self.p_vy[k] = vel[1] + thermal * z1
        self.p_internal[k] = 0.5 * ti.max(internal_dof, 0.0) * theta
        self.p_mass[k] = ti.exp(-self.dt / self.tau[i, j]) * rho / ti.cast(self.particles_per_cell, ti.f64)
        self.p_tfree[k] = self.dt
        self.p_alive[k] = 1

    @ti.func
    def sample_cell_particles(self, i, j, rho, vel, pressure):
        for p in range(self.particles_per_cell):
            k = self.particle_index(i, j, p)
            self.sample_particle_in_cell(k, i, j, rho, vel, pressure)

    @ti.func
    def protected_w(self, w):
        rho = ti.max(w[0], 1.0e-8)
        mom_x = w[1]
        mom_y = w[2]
        min_energy = 0.5 * (mom_x * mom_x + mom_y * mom_y) / rho + 1.0e-8
        return ti.Vector([rho, mom_x, mom_y, ti.max(w[3], min_energy)])

    @ti.func
    def conservative_w(self, i, j):
        return ti.Vector(
            [
                self.rho[i, j],
                self.rho[i, j] * self.u[i, j][0],
                self.rho[i, j] * self.u[i, j][1],
                self.rhoE[i, j],
            ]
        )

    @ti.func
    def sod_left_w(self):
        return self.conservative_from_primitive(1.0, ti.Vector([0.0, 0.0]), 1.0)

    @ti.func
    def sod_right_w(self):
        return self.conservative_from_primitive(0.125, ti.Vector([0.0, 0.0]), 0.1)

    @ti.func
    def minmod_w(self, left, right):
        out = ti.Vector.zero(ti.f64, 4)
        for c in ti.static(range(4)):
            if left[c] * right[c] > 0.0:
                out[c] = ti.select(ti.abs(left[c]) < ti.abs(right[c]), left[c], right[c])
        return out

    @ti.func
    def total_gradients(self, i, j):
        jp = self.wrap_j(j + 1)
        jm = self.wrap_j(j - 1)
        w = self.conservative_w(i, j)
        w_left = self.conservative_w(self.wrap_i(i - 1), j)
        w_right = self.conservative_w(self.wrap_i(i + 1), j)
        if self.fixed_x_boundaries[None] == 1:
            if i == 0:
                w_left = self.sod_left_w()
            if i == self.nx - 1:
                w_right = self.sod_right_w()
        grad_x = self.minmod_w(w - w_left, w_right - w)
        grad_y = self.minmod_w(w - self.conservative_w(i, jm), self.conservative_w(i, jp) - w)
        return grad_x, grad_y

    @ti.func
    def hidden_gradients(self, i, j):
        jp = self.wrap_j(j + 1)
        jm = self.wrap_j(j - 1)
        w = self.wh[i, j]
        w_left = self.wh[self.wrap_i(i - 1), j]
        w_right = self.wh[self.wrap_i(i + 1), j]
        if self.fixed_x_boundaries[None] == 1:
            if i == 0:
                w_left = self.sod_left_w()
            if i == self.nx - 1:
                w_right = self.sod_right_w()
        grad_x = self.minmod_w(w - w_left, w_right - w)
        grad_y = self.minmod_w(w - self.wh[i, jm], self.wh[i, jp] - w)
        return grad_x, grad_y

    @ti.func
    def rotate_w(self, w):
        return ti.Vector([w[0], w[2], w[1], w[3]])

    @ti.func
    def unrotate_flux(self, flux):
        return ti.Vector([flux[0], flux[2], flux[1], flux[3]])

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
    def half_normal_moments(self, velocity, theta):
        thermal = ti.sqrt(ti.max(theta, 1.0e-8))
        z = velocity / thermal
        phi = 0.3989422804014327 * ti.exp(-0.5 * z * z)
        h0 = 0.5 * (1.0 + self.erf_approx(0.7071067811865476 * z))
        h1 = velocity * h0 + thermal * phi
        h2 = velocity * h1 + theta * h0
        return ti.Vector([h0, h1, h2])

    @ti.func
    def half_range_interface_w(self, wl, wr):
        wl = self.protected_w(wl)
        wr = self.protected_w(wr)
        rho_l, vel_l, p_l = self.primitive_from_conservative(wl)
        rho_r, vel_r, p_r = self.primitive_from_conservative(wr)
        theta_l = p_l / rho_l
        theta_r = p_r / rho_r
        h_l = self.half_normal_moments(vel_l[0], theta_l)
        h_r = self.half_normal_moments(vel_r[0], theta_r)
        internal_dof = 2.0 / (ti.static(self.gamma) - 1.0) - 2.0

        rho = rho_l * h_l[0] + rho_r * (1.0 - h_r[0])
        mom_n = rho_l * h_l[1] + rho_r * (vel_r[0] - h_r[1])
        mom_t = rho_l * vel_l[1] * h_l[0] + rho_r * vel_r[1] * (1.0 - h_r[0])
        energy_l = 0.5 * rho_l * (h_l[2] + h_l[0] * (vel_l[1] * vel_l[1] + (1.0 + internal_dof) * theta_l))
        energy_r_pos = 0.5 * rho_r * (h_r[2] + h_r[0] * (vel_r[1] * vel_r[1] + (1.0 + internal_dof) * theta_r))
        energy = energy_l + wr[3] - energy_r_pos
        return self.protected_w(ti.Vector([rho, mom_n, mom_t, energy]))

    @ti.func
    def second_normal_moment(self, w):
        w = self.protected_w(w)
        rho, vel, pressure = self.primitive_from_conservative(w)
        theta = pressure / rho
        internal_dof = 2.0 / (ti.static(self.gamma) - 1.0) - 2.0
        un = vel[0]
        ut = vel[1]
        un2 = un * un + theta
        return ti.Vector(
            [
                rho * un2,
                rho * (un * un * un + 3.0 * un * theta),
                rho * ut * un2,
                0.5 * rho * (
                    un * un * un * un
                    + 6.0 * un * un * theta
                    + 3.0 * theta * theta
                    + un2 * (ut * ut + (1.0 + internal_dof) * theta)
                ),
            ]
        )

    @ti.func
    def normal_tangential_moment(self, w):
        w = self.protected_w(w)
        rho, vel, pressure = self.primitive_from_conservative(w)
        theta = pressure / rho
        internal_dof = 2.0 / (ti.static(self.gamma) - 1.0) - 2.0
        un = vel[0]
        ut = vel[1]
        return ti.Vector(
            [
                rho * un * ut,
                rho * (un * un + theta) * ut,
                rho * un * (ut * ut + theta),
                0.5 * rho * un * ut * (un * un + ut * ut + (6.0 + internal_dof) * theta),
            ]
        )

    @ti.func
    def derivative_flux_normal(self, w, direction):
        epsilon = 1.0e-4
        return (self.euler_flux_x_from_w(self.protected_w(w + epsilon * direction)) - self.euler_flux_x_from_w(self.protected_w(w - epsilon * direction))) / (2.0 * epsilon)

    @ti.func
    def derivative_flux_tangent(self, w, direction):
        epsilon = 1.0e-4
        return (self.euler_flux_y_from_w(self.protected_w(w + epsilon * direction)) - self.euler_flux_y_from_w(self.protected_w(w - epsilon * direction))) / (2.0 * epsilon)

    @ti.func
    def derivative_second_normal(self, w, direction):
        epsilon = 1.0e-4
        return (self.second_normal_moment(self.protected_w(w + epsilon * direction)) - self.second_normal_moment(self.protected_w(w - epsilon * direction))) / (2.0 * epsilon)

    @ti.func
    def derivative_normal_tangential(self, w, direction):
        epsilon = 1.0e-4
        return (self.normal_tangential_moment(self.protected_w(w + epsilon * direction)) - self.normal_tangential_moment(self.protected_w(w - epsilon * direction))) / (2.0 * epsilon)

    @ti.func
    def analytic_wave_flux_parts(self, w0, grad_n, grad_t, wh0, grad_h_n, grad_h_t):
        _, tau = self.relaxation_from_w(w0)
        dt = ti.static(self.dt)
        decay = ti.exp(-dt / tau)
        c1 = 1.0 - tau * (1.0 - decay) / dt
        # c2 = (2.0 * tau * tau * (1.0 - decay) - tau * dt * decay - tau * dt) / dt
        c2 = -tau + 2.0 * tau * tau / dt - decay * (tau + 2.0 * tau * tau / dt)
        c3 = 0.5 * dt - tau + tau * tau * (1.0 - decay) / dt
        c4 = tau * (1.0 - decay) / dt
        c5 = tau * decay - tau * tau * (1.0 - decay) / dt
        c6 = decay
        c7 = 0.5 * dt * decay

        flux_g0 = self.euler_flux_x_from_w(w0)
        grad_transport_g0 = self.derivative_second_normal(w0, grad_n) + self.derivative_normal_tangential(w0, grad_t)
        time_gradient_w0 = -(self.derivative_flux_normal(w0, grad_n) + self.derivative_flux_tangent(w0, grad_t))
        time_transport_g0 = self.derivative_flux_normal(w0, time_gradient_w0)
        flux_eq = c1 * flux_g0 + c2 * grad_transport_g0 + c3 * time_transport_g0

        # Liu et al. use the permitted first-order approximation g+ = g.
        flux_gh = self.euler_flux_x_from_w(wh0)
        grad_transport_gh = self.derivative_second_normal(wh0, grad_h_n) + self.derivative_normal_tangential(wh0, grad_h_t)
        flux_fr_wave = c4 * flux_g0 + c5 * grad_transport_g0 - c6 * flux_gh + c7 * grad_transport_gh
        return flux_eq, flux_fr_wave

    @ti.func
    def ugkwp_face_flux_x(self, i, j, ip):
        grad_l_x, grad_l_y = self.total_gradients(i, j)
        grad_r_x, grad_r_y = self.total_gradients(ip, j)
        grad_h_l_x, grad_h_l_y = self.hidden_gradients(i, j)
        grad_h_r_x, grad_h_r_y = self.hidden_gradients(ip, j)
        wl = self.protected_w(self.conservative_w(i, j) + 0.5 * grad_l_x)
        wr = self.protected_w(self.conservative_w(ip, j) - 0.5 * grad_r_x)
        wh_l = self.protected_w(self.wh[i, j] + 0.5 * grad_h_l_x)
        wh_r = self.protected_w(self.wh[ip, j] - 0.5 * grad_h_r_x)
        flux_eq, flux_fr_wave = self.analytic_wave_flux_parts(
            self.half_range_interface_w(wl, wr),
            0.5 * (grad_l_x + grad_r_x),
            0.5 * (grad_l_y + grad_r_y),
            self.half_range_interface_w(wh_l, wh_r),
            0.5 * (grad_h_l_x + grad_h_r_x),
            0.5 * (grad_h_l_y + grad_h_r_y),
        )
        return flux_eq + flux_fr_wave

    @ti.func
    def ugkwp_face_flux_y(self, i, j, jp):
        grad_l_x, grad_l_y = self.total_gradients(i, j)
        grad_r_x, grad_r_y = self.total_gradients(i, jp)
        grad_h_l_x, grad_h_l_y = self.hidden_gradients(i, j)
        grad_h_r_x, grad_h_r_y = self.hidden_gradients(i, jp)
        wl = self.rotate_w(self.protected_w(self.conservative_w(i, j) + 0.5 * grad_l_y))
        wr = self.rotate_w(self.protected_w(self.conservative_w(i, jp) - 0.5 * grad_r_y))
        wh_l = self.rotate_w(self.protected_w(self.wh[i, j] + 0.5 * grad_h_l_y))
        wh_r = self.rotate_w(self.protected_w(self.wh[i, jp] - 0.5 * grad_h_r_y))
        flux_eq, flux_fr_wave = self.analytic_wave_flux_parts(
            self.half_range_interface_w(wl, wr),
            0.5 * (self.rotate_w(grad_l_y) + self.rotate_w(grad_r_y)),
            0.5 * (self.rotate_w(grad_l_x) + self.rotate_w(grad_r_x)),
            self.half_range_interface_w(wh_l, wh_r),
            0.5 * (self.rotate_w(grad_h_l_y) + self.rotate_w(grad_h_r_y)),
            0.5 * (self.rotate_w(grad_h_l_x) + self.rotate_w(grad_h_r_x)),
        )
        return self.unrotate_flux(flux_eq + flux_fr_wave)

    @ti.func
    def ugkwp_boundary_flux_x(self, wl, wr, wh_l, wh_r):
        """First-order reservoir face flux; gradients vanish in the ghost cell."""
        zero = ti.Vector.zero(ti.f64, 4)
        flux_eq, flux_fr_wave = self.analytic_wave_flux_parts(
            self.half_range_interface_w(wl, wr),
            zero,
            zero,
            self.half_range_interface_w(wh_l, wh_r),
            zero,
            zero,
        )
        return flux_eq + flux_fr_wave

    @ti.kernel
    def init(self):
        for i, j in self.rho:
            if i == 0 and j == 0:
                self.fixed_x_boundaries[None] = 0
            y = (ti.cast(j, ti.f64) + 0.5) / ti.cast(self.ny, ti.f64)
            amp = self.mach * ti.static(self.cs)
            ux = amp * ti.sin(2.0 * math.pi * y)
            vel = ti.Vector([ux, 0.0])
            pressure = 1.0 / ti.static(self.gamma)
            w = self.conservative_from_primitive(1.0, vel, pressure)
            self.rho[i, j] = w[0]
            self.rhoE[i, j] = w[3]
            self.u[i, j] = vel
            self.update_cell_relaxation(i, j, w)
            self.sample_cell_particles(i, j, w[0], vel, pressure)

    @ti.kernel
    def init_periodic_sod(self):
        """Initialize the Sod states and enable fixed reservoir boundaries in x."""
        for i, j in self.rho:
            if i == 0 and j == 0:
                self.fixed_x_boundaries[None] = 1
            x = (ti.cast(i, ti.f64) + 0.5) / ti.cast(self.nx, ti.f64) - 0.5
            rho = 0.125
            pressure = 0.1
            if x >= 0.0:
                rho = 1.0
                pressure = 1.0
            vel = ti.Vector([0.0, 0.0])
            w = self.conservative_from_primitive(rho, vel, pressure)
            self.rho[i, j] = w[0]
            self.rhoE[i, j] = w[3]
            self.u[i, j] = vel
            self.update_cell_relaxation(i, j, w)
            self.sample_cell_particles(i, j, rho, vel, pressure)

    @ti.kernel
    def update_relaxation(self):
        for i, j in self.rho:
            self.update_cell_relaxation(i, j, self.conservative_w(i, j))

    @ti.kernel
    def compute_face_fluxes_x(self):
        for i, j in self.rho:
            if self.fixed_x_boundaries[None] == 1:
                if i == 0:
                    self.face_x_left[j] = self.ugkwp_boundary_flux_x(
                        self.sod_left_w(), self.conservative_w(0, j), self.sod_left_w(), self.wh[0, j]
                    )
                if i < self.nx - 1:
                    self.face_x[i, j] = self.ugkwp_face_flux_x(i, j, i + 1)
                else:
                    self.face_x[i, j] = self.ugkwp_boundary_flux_x(
                        self.conservative_w(i, j), self.sod_right_w(), self.wh[i, j], self.sod_right_w()
                    )
            else:
                ip = self.wrap_i(i + 1)
                self.face_x[i, j] = self.ugkwp_face_flux_x(i, j, ip)

    @ti.kernel
    def compute_face_fluxes_y(self):
        for i, j in self.rho:
            jp = self.wrap_j(j + 1)
            self.face_y[i, j] = self.ugkwp_face_flux_y(i, j, jp)

    def compute_face_fluxes(self):
        self.compute_face_fluxes_x()
        self.compute_face_fluxes_y()
        self.transport_particles()

    @ti.kernel
    def clear_particle_moments(self):
        for i, j in self.rho:
            self.wp[i, j] = ti.Vector([0.0, 0.0, 0.0, 0.0])

    @ti.kernel
    def accumulate_particle_moments(self):
        for k in range(self.max_particles):
            if self.p_alive[k] == 1:
                i = self.wrap_i(ti.cast(ti.floor(self.p_x[k]), ti.i32))
                j = self.wrap_j(ti.cast(ti.floor(self.p_y[k]), ti.i32))
                mass = self.p_mass[k]
                e = self.particle_energy(k)
                ti.atomic_add(self.wp[i, j][0], mass)
                ti.atomic_add(self.wp[i, j][1], mass * self.p_vx[k])
                ti.atomic_add(self.wp[i, j][2], mass * self.p_vy[k])
                ti.atomic_add(self.wp[i, j][3], e)

    @ti.kernel
    def compute_wh(self):
        for i, j in self.rho:
            w = ti.Vector(
                [
                    self.rho[i, j],
                    self.rho[i, j] * self.u[i, j][0],
                    self.rho[i, j] * self.u[i, j][1],
                    self.rhoE[i, j],
                ]
            )
            wh = w - self.wp[i, j]
            if self.fixed_x_boundaries[None] == 1:
                if i == 0:
                    wh = self.sod_left_w()
                elif i == self.nx - 1:
                    wh = self.sod_right_w()
            # rho = ti.max(wh[0], 1.0e-8)
            # vel = ti.Vector([wh[1] / rho, wh[2] / rho])
            # kinetic = 0.5 * rho * (vel[0] * vel[0] + vel[1] * vel[1])
            # pressure = ti.max((ti.static(self.gamma) - 1.0) * (wh[3] - kinetic), 1.0e-8)

            # wh_raw = w - self.wp[i, j]
            # self.wh[i, j] = wh_raw

            # wh_safe = make_positive_state(wh_raw)
            # self.wh_safe[i, j] = wh_safe
            # wh[0] = rho
            # wh[1] = rho * vel[0]
            # wh[2] = rho * vel[1]
            # wh[3] = pressure / (ti.static(self.gamma) - 1.0) + kinetic
            self.wh[i, j] = wh

    def update_particle_moments(self):
        self.clear_particle_moments()
        self.accumulate_particle_moments()
        self.compute_wh()

    @ti.kernel
    def transport_particles(self):
        for k in range(self.max_particles):
            x0 = self.p_x[k]
            y0 = self.p_y[k]
            vx = self.p_vx[k]
            vy = self.p_vy[k]
            mass = self.p_mass[k]
            free_t = 0.0
            if self.p_alive[k] == 1:
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

            if self.p_alive[k] == 1 and cross_tx <= free_t:
                scale = sign_x * mass / ti.static(self.dt)
                if self.fixed_x_boundaries[None] == 1 and i == 0 and sign_x < 0.0:
                    ti.atomic_add(self.face_x_left[j][0], scale)
                    ti.atomic_add(self.face_x_left[j][1], scale * vx)
                    ti.atomic_add(self.face_x_left[j][2], scale * vy)
                    ti.atomic_add(self.face_x_left[j][3], sign_x * self.particle_energy(k) / ti.static(self.dt))
                else:
                    ti.atomic_add(self.face_x[face_i, j][0], scale)
                    ti.atomic_add(self.face_x[face_i, j][1], scale * vx)
                    ti.atomic_add(self.face_x[face_i, j][2], scale * vy)
                    ti.atomic_add(self.face_x[face_i, j][3], sign_x * self.particle_energy(k) / ti.static(self.dt))

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

            if self.p_alive[k] == 1 and cross_ty <= free_t:
                scale = sign_y * mass / ti.static(self.dt)
                ti.atomic_add(self.face_y[i, face_j][0], scale)
                ti.atomic_add(self.face_y[i, face_j][1], scale * vx)
                ti.atomic_add(self.face_y[i, face_j][2], scale * vy)
                ti.atomic_add(self.face_y[i, face_j][3], sign_y * self.particle_energy(k) / ti.static(self.dt))

            x1 = x0 + vx * free_t
            y1 = y0 + vy * free_t
            left_domain = 0
            if self.fixed_x_boundaries[None] == 1:
                if x1 < 0.0:
                    x1 = 0.5
                    left_domain = 1
                elif x1 >= ti.cast(self.nx, ti.f64):
                    x1 = ti.cast(self.nx, ti.f64) - 0.5
                    left_domain = 1
            else:
                if x1 < 0.0:
                    x1 += ti.cast(self.nx, ti.f64)
                elif x1 >= ti.cast(self.nx, ti.f64):
                    x1 -= ti.cast(self.nx, ti.f64)
            if y1 < 0.0:
                y1 += ti.cast(self.ny, ti.f64)
            elif y1 >= ti.cast(self.ny, ti.f64):
                y1 -= ti.cast(self.ny, ti.f64)
            self.p_x[k] = x1
            self.p_y[k] = y1
            if self.p_alive[k] == 1:
                if self.p_tfree[k] < ti.static(self.dt) or left_domain == 1:
                    self.p_alive[k] = 0
                    self.p_mass[k] = 0.0
                else:
                    self.p_tfree[k] -= ti.static(self.dt)

    @ti.kernel
    def update(self):
        for i, j in self.rho:
            jm = self.wrap_j(j - 1)
            left_x_flux = self.face_x[self.wrap_i(i - 1), j]
            if self.fixed_x_boundaries[None] == 1 and i == 0:
                left_x_flux = self.face_x_left[j]
            div = (self.face_x[i, j] - left_x_flux) + (self.face_y[i, j] - self.face_y[i, jm])
            rho_old = self.rho[i, j]
            mom_old = rho_old * self.u[i, j]
            energy_old = self.rhoE[i, j]
            rho_new = ti.max(rho_old - self.dt * div[0], 1.0e-8)
            mom_new = mom_old - self.dt * ti.Vector([div[1], div[2]])
            energy_new = ti.max(energy_old - self.dt * div[3], 1.0e-8)
            if self.fixed_x_boundaries[None] == 1:
                if i == 0:
                    fixed_w = self.sod_left_w()
                    rho_new = fixed_w[0]
                    mom_new = ti.Vector([fixed_w[1], fixed_w[2]])
                    energy_new = fixed_w[3]
                elif i == self.nx - 1:
                    fixed_w = self.sod_right_w()
                    rho_new = fixed_w[0]
                    mom_new = ti.Vector([fixed_w[1], fixed_w[2]])
                    energy_new = fixed_w[3]
            self.rho_next[i, j] = rho_new
            self.rhoE_next[i, j] = energy_new
            self.u_next[i, j] = mom_new / rho_new

    @ti.kernel
    def swap(self):
        for i, j in self.rho:
            self.rho[i, j] = self.rho_next[i, j]
            self.rhoE[i, j] = self.rhoE_next[i, j]
            self.u[i, j] = self.u_next[i, j]

    @ti.kernel
    def refresh_particles(self):
        for k in range(self.max_particles):
            i = self.wrap_i(ti.cast(ti.floor(self.p_x[k]), ti.i32))
            j = self.wrap_j(ti.cast(ti.floor(self.p_y[k]), ti.i32))
            if self.p_alive[k] == 1:
                if self.p_tfree[k] <= 1.0e-12:
                    self.p_tfree[k] = -self.tau[i, j] * ti.log(ti.max(ti.random(ti.f64), 1.0e-12))
            else:
                rho_h, vel_h, pressure_h = self.primitive_from_conservative(self.wh[i, j])
                self.sample_particle_in_cell(k, i, j, rho_h, vel_h, pressure_h)

    def step(self):
        self.update_relaxation()
        self.update_particle_moments()
        self.compute_face_fluxes()
        self.update()
        self.swap()
        self.update_relaxation()
        self.update_particle_moments()
        self.refresh_particles()

    def kinetic_energy(self):
        rho_np = self.rho.to_numpy()
        u_np = self.u.to_numpy()
        return float(0.5 * np.mean(rho_np * np.sum(u_np * u_np, axis=-1)))

    def raw_pressure_array(self):
        rho_np = self.rho.to_numpy()
        u_np = self.u.to_numpy()
        rhoE_np = self.rhoE.to_numpy()
        kinetic = 0.5 * rho_np * np.sum(u_np * u_np, axis=-1)
        return (self.gamma - 1.0) * (rhoE_np - kinetic)

    def pressure_array(self):
        return np.maximum(self.raw_pressure_array(), 1.0e-12)

    def pressure_diagnostics(self):
        pressure_raw = self.raw_pressure_array()
        return float(np.min(pressure_raw)), int(np.count_nonzero(pressure_raw < 0.0))

    def density_stats(self):
        rho_np = self.rho.to_numpy()
        return float(rho_np.min()), float(rho_np.mean()), float(rho_np.max())

    def particle_stats(self):
        alive_np = self.p_alive.to_numpy()
        mass_np = self.p_mass.to_numpy()
        alive = int(np.sum(alive_np))
        total_mass = float(np.sum(mass_np))
        alive_mass = float(np.sum(mass_np[alive_np == 1]))
        return alive, alive_mass, total_mass

    def snapshot_arrays(self):
        return self.u.to_numpy()[:, :, 0]

    def sod_profile(self):
        rho_np = self.rho.to_numpy()
        u_np = self.u.to_numpy()
        pressure_np = self.pressure_array()
        x = (np.arange(self.nx, dtype=np.float32) + 0.5) / float(self.nx) - 0.5
        rho = np.mean(rho_np, axis=1)
        ux = np.mean(u_np[:, :, 0], axis=1)
        pressure = np.mean(pressure_np, axis=1)
        temperature = pressure / np.maximum(rho, 1.0e-12)
        return x, rho, ux, pressure, temperature

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

    def render_density_image(self):
        values = self.rho.to_numpy()
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        normalized = np.clip((values - vmin) / max(vmax - vmin, 1.0e-12), 0.0, 1.0)
        image = np.zeros((self.nx, self.ny, 3), dtype=np.float32)
        image[:, :, 0] = normalized
        image[:, :, 1] = 0.15 + 0.7 * (1.0 - np.abs(2.0 * normalized - 1.0))
        image[:, :, 2] = 1.0 - normalized
        return image

    def save_results(self, output_dir, tag, benchmark, time):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rho_np = self.rho.to_numpy()
        vel_np = self.u.to_numpy()
        rhoE_np = self.rhoE.to_numpy()
        raw_pressure_np = self.raw_pressure_array()
        internal_energy_np = raw_pressure_np / (self.gamma - 1.0)
        npz_path = output_dir / f"{tag}_rho_vel.npz"
        plot_path = output_dir / f"{tag}_profiles.png"
        np.savez(
            npz_path,
            rho=rho_np,
            vel=vel_np,
            rhoE=rhoE_np,
            pressure=self.pressure_array(),
            pressure_raw=raw_pressure_np,
            internal_energy=internal_energy_np,
        )
        self.save_profile_plot(plot_path, benchmark, time)
        return npz_path, plot_path

    def save_profile_plot(self, path, benchmark, time):
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x, rho, ux, pressure, temperature = self.sod_profile()
        fig, axes = plt.subplots(4, 1, figsize=(7.0, 9.0), sharex=True)
        fig.suptitle(f"{benchmark} UGKWP profiles, t = {time:.4f}")

        axes[0].plot(x, rho, "o-", markersize=2.5, linewidth=1.0)
        axes[0].set_ylabel(r"$\rho$")
        axes[0].grid(True, alpha=0.25)

        axes[1].plot(x, ux, "o-", markersize=2.5, linewidth=1.0)
        axes[1].set_ylabel(r"$u_x$")
        axes[1].grid(True, alpha=0.25)

        axes[2].plot(x, pressure, "o-", markersize=2.5, linewidth=1.0)
        axes[2].set_ylabel(r"$p$")
        axes[2].grid(True, alpha=0.25)

        axes[3].plot(x, temperature, "o-", markersize=2.5, linewidth=1.0)
        axes[3].set_ylabel(r"$T$")
        axes[3].set_xlabel("x")
        axes[3].grid(True, alpha=0.25)

        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)


SinglePhaseGKS2D = SinglePhaseUGKWP2D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("shear", "sod"), default="sod")
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=256)
    parser.add_argument("--particles-per-cell", type=int, default=300)
    parser.add_argument("--kn", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--t-end", type=float, default=0.15)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--save-prefix", default=None)
    args = parser.parse_args()

    ti.init(arch=ti.gpu, default_fp=ti.f64, random_seed=args.seed)
    print(f"random seed: {args.seed}")
    sim = SinglePhaseUGKWP2D(args.nx, args.ny, args.particles_per_cell, args.kn)
    if args.benchmark == "sod":
        sim.init_periodic_sod()
        title = "Fixed-boundary Sod UGKWP"
    else:
        sim.init()
        title = "BGK-UGKWP"
    gui = ti.GUI(title, res=(sim.nx, sim.ny), fast_gui=False)
    s = 0
    sim_time = 0.0
    while gui.running and (args.benchmark != "sod" or sim_time < args.t_end):
        # print(s, sim_time)
        sim.step()
        if args.benchmark == "sod":
            gui.set_image(sim.render_density_image())
        else:
            gui.set_image(sim.render_image())
        gui.show()
        s += 1
        sim_time += sim.dt
        if args.benchmark == "sod" and s % 10 == 0:
            rho_min, rho_mean, rho_max = sim.density_stats()
            p_raw_min, negative_pressure_cells = sim.pressure_diagnostics()
            print(
                f"t={sim_time:.4f} rho_min={rho_min:.6e} rho_mean={rho_mean:.6e} rho_max={rho_max:.6e} "
                f"p_raw_min={p_raw_min:.6e} p_raw_negative_cells={negative_pressure_cells}"
            )
        elif s % 100 == 0:
            l2, linf, amp_num, amp_exact = sim.shear_wave_error(sim_time)
            print(f" l2={l2:.6e} linf={linf:.6e} amp_num={amp_num:.6e} amp_exact={amp_exact:.6e}")

    if args.benchmark == "sod":
        x, rho, ux, pressure, temperature = sim.sod_profile()
        p_raw_min, negative_pressure_cells = sim.pressure_diagnostics()
        mid = sim.nx // 2
        print(
            "fixed-boundary Sod profile sample: "
            f"x={x[mid]:.6e} rho={rho[mid]:.6e} ux={ux[mid]:.6e} p={pressure[mid]:.6e} T={temperature[mid]:.6e}"
        )
        print(f"raw pressure diagnostic: min={p_raw_min:.6e} negative_cells={negative_pressure_cells}")

    tag = args.save_prefix
    if tag is None:
        tag = f"{args.benchmark}_nx{args.nx}_ny{args.ny}_t{sim_time:.3f}".replace(".", "p")
    npz_path, plot_path = sim.save_results(args.output_dir, tag, args.benchmark, sim_time)
    print(f"saved rho/vel arrays: {npz_path}")
    print(f"saved profile plot: {plot_path}")


if __name__ == "__main__":
    main()
