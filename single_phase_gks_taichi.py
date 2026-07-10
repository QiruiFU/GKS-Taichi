import math
import numpy as np
import taichi as ti


@ti.data_oriented
class SinglePhaseGKS2D:
    """A small teaching implementation of 2D isothermal single-phase GKS.

    Stored variables:
      rho(x, y), u(x, y)

    Equation of state:
      p = rho * cs^2

    The face flux is reconstructed from a local BGK/GKS expansion:
      f ~= f_eq - tau * D f_eq + (dt / 2) * d_t f_eq

    This is intentionally much smaller than the two-phase paper code:
      no phase field, no surface tension, no psi = p - rho cs^2,
      no Allen-Cahn regularization, no independent pressure evolution.
    """

    def __init__(self):
        self.nx = 256
        self.ny = 256
        self.dt = 0.01
        self.cs = 1.0 / math.sqrt(3.0)
        self.cs2 = self.cs * self.cs
        self.nu = 0.05
        self.mach = 0.3
        self.tau = self.nu / self.cs2

        self.rho = ti.field(ti.f32, shape=(self.nx, self.ny))
        self.rho_next = ti.field(ti.f32, shape=(self.nx, self.ny))
        self.u = ti.Vector.field(2, ti.f32, shape=(self.nx, self.ny))
        self.u_next = ti.Vector.field(2, ti.f32, shape=(self.nx, self.ny))

        # face_x[i, j] is the flux through the face between (i, j) and (i+1, j).
        # face_y[i, j] is the flux through the face between (i, j) and (i, j+1).
        # Each flux stores [mass, x-momentum, y-momentum].
        self.face_x = ti.Vector.field(3, ti.f32, shape=(self.nx, self.ny))
        self.face_y = ti.Vector.field(3, ti.f32, shape=(self.nx, self.ny))

    @ti.func
    def wrap_i(self, i):
        return (i + self.nx) % self.nx

    @ti.func
    def wrap_j(self, j):
        return (j + self.ny) % self.ny

    @ti.func
    def moment_1d(self, u, k):
        cs2 = ti.static(self.cs2)
        cs4 = cs2 * cs2
        out = 0.0
        if k == 0:
            out = 1.0
        elif k == 1:
            out = u
        elif k == 2:
            out = u * u + cs2
        elif k == 3:
            out = u * u * u + 3.0 * cs2 * u
        elif k == 4:
            out = u**4 + 6.0 * cs2 * u * u + 3.0 * cs4
        return out

    @ti.func
    def moment(self, vel, ix, iy):
        return self.moment_1d(vel[0], ix) * self.moment_1d(vel[1], iy)

    @ti.func
    def solve_coefficients(self, h, vel):
        # If h = <[1, xi_x, xi_y] * (a0 + a1 xi_x + a2 xi_y) Gamma_u>,
        # then recover a = [a0, a1, a2].
        cs2_inv = ti.static(1.0 / self.cs2)
        a = ti.Vector([0.0, 0.0, 0.0])
        a[1] = cs2_inv * (h[1] - vel[0] * h[0])
        a[2] = cs2_inv * (h[2] - vel[1] * h[0])
        a[0] = h[0] - vel[0] * a[1] - vel[1] * a[2]
        return a

    @ti.func
    def gks_flux_local(self, rho_f, vel, grad_rho, grad_u, component):
        """Flux through a local +x face.

        component:
          0 -> mass flux
          1 -> local x-momentum flux
          2 -> local y-momentum flux
        """
        cs2_inv = ti.static(1.0 / self.cs2)

        # Spatial derivative coefficients of f_eq = rho * Gamma_u.
        # d_x f_eq = (a0 + a1 xi_x + a2 xi_y) Gamma_u
        # d_y f_eq = (b0 + b1 xi_x + b2 xi_y) Gamma_u
        a = ti.Vector([0.0, 0.0, 0.0])
        b = ti.Vector([0.0, 0.0, 0.0])

        ux_x = grad_u[0, 0]
        uy_x = grad_u[1, 0]
        ux_y = grad_u[0, 1]
        uy_y = grad_u[1, 1]

        a[0] = grad_rho[0] - rho_f * (vel[0] * ux_x + vel[1] * uy_x) * cs2_inv
        a[1] = rho_f * ux_x * cs2_inv
        a[2] = rho_f * uy_x * cs2_inv

        b[0] = grad_rho[1] - rho_f * (vel[0] * ux_y + vel[1] * uy_y) * cs2_inv
        b[1] = rho_f * ux_y * cs2_inv
        b[2] = rho_f * uy_y * cs2_inv

        # Temporal derivative coefficients A from compatibility:
        # <D f_eq> = 0, <xi D f_eq> = 0, without body force in the flux.
        h = ti.Vector([0.0, 0.0, 0.0])
        for c in ti.static(range(3)):
            rx = 0
            ry = 0
            if c == 1:
                rx = 1
            elif c == 2:
                ry = 1

            s = (
                a[0] * self.moment(vel, 1 + rx, ry)
                + b[0] * self.moment(vel, rx, 1 + ry)
                + a[1] * self.moment(vel, 2 + rx, ry)
                + b[2] * self.moment(vel, rx, 2 + ry)
                + (a[2] + b[1]) * self.moment(vel, 1 + rx, 1 + ry)
            )
            h[c] = -s
        A = self.solve_coefficients(h, vel)

        rx = 0
        ry = 0
        if component == 1:
            rx = 1
        elif component == 2:
            ry = 1

        geq = rho_f * self.moment(vel, 1 + rx, ry)

        spatial = (
            a[0] * self.moment(vel, 2 + rx, ry)
            + b[0] * self.moment(vel, 1 + rx, 1 + ry)
            + a[1] * self.moment(vel, 3 + rx, ry)
            + b[2] * self.moment(vel, 1 + rx, 2 + ry)
            + (a[2] + b[1]) * self.moment(vel, 2 + rx, 1 + ry)
        )

        temporal = (
            A[0] * self.moment(vel, 1 + rx, ry)
            + A[1] * self.moment(vel, 2 + rx, ry)
            + A[2] * self.moment(vel, 1 + rx, 1 + ry)
        )

        return geq - self.tau * spatial + (0.5 * self.dt - self.tau) * temporal

    @ti.kernel
    def init(self):
        for i, j in self.rho:
            y = (ti.cast(j, ti.f32) + 0.5) / ti.cast(self.ny, ti.f32)
            amp = self.mach * ti.static(self.cs)
            ux = amp * ti.sin(2.0 * math.pi * y)
            self.rho[i, j] = 1.0
            self.u[i, j] = ti.Vector([ux, 0.0])

    @ti.kernel
    def compute_face_fluxes(self):
        for i, j in self.rho:
            ip = self.wrap_i(i + 1)
            im = self.wrap_i(i - 1)
            jp = self.wrap_j(j + 1)
            jm = self.wrap_j(j - 1)

            # x-normal face between (i, j) and (i+1, j).
            rho_f = 0.5 * (self.rho[i, j] + self.rho[ip, j])
            vel = 0.5 * (self.u[i, j] + self.u[ip, j])
            grad_rho = ti.Vector(
                [
                    self.rho[ip, j] - self.rho[i, j],
                    0.25 * ((self.rho[i, jp] + self.rho[ip, jp]) - (self.rho[i, jm] + self.rho[ip, jm])),
                ]
            )
            grad_u = ti.Matrix.zero(ti.f32, 2, 2)
            du_dx = self.u[ip, j] - self.u[i, j]
            du_dy = 0.25 * ((self.u[i, jp] + self.u[ip, jp]) - (self.u[i, jm] + self.u[ip, jm]))
            grad_u[0, 0] = du_dx[0]
            grad_u[1, 0] = du_dx[1]
            grad_u[0, 1] = du_dy[0]
            grad_u[1, 1] = du_dy[1]
            self.face_x[i, j] = ti.Vector(
                [
                    self.gks_flux_local(rho_f, vel, grad_rho, grad_u, 0),
                    self.gks_flux_local(rho_f, vel, grad_rho, grad_u, 1),
                    self.gks_flux_local(rho_f, vel, grad_rho, grad_u, 2),
                ]
            )

            # y-normal face between (i, j) and (i, j+1).
            # Rotate world coordinates to local coordinates:
            # local x = world y, local y = world x.
            rho_f_y = 0.5 * (self.rho[i, j] + self.rho[i, jp])
            vel_w = 0.5 * (self.u[i, j] + self.u[i, jp])
            vel_l = ti.Vector([vel_w[1], vel_w[0]])

            grad_rho_l = ti.Vector(
                [
                    self.rho[i, jp] - self.rho[i, j],
                    0.25 * ((self.rho[ip, j] + self.rho[ip, jp]) - (self.rho[im, j] + self.rho[im, jp])),
                ]
            )

            du_dlocal_x_w = self.u[i, jp] - self.u[i, j]
            du_dlocal_y_w = 0.25 * ((self.u[ip, j] + self.u[ip, jp]) - (self.u[im, j] + self.u[im, jp]))
            grad_u_l = ti.Matrix.zero(ti.f32, 2, 2)
            # local velocity components are [world_y, world_x].
            grad_u_l[0, 0] = du_dlocal_x_w[1]
            grad_u_l[1, 0] = du_dlocal_x_w[0]
            grad_u_l[0, 1] = du_dlocal_y_w[1]
            grad_u_l[1, 1] = du_dlocal_y_w[0]

            f0 = self.gks_flux_local(rho_f_y, vel_l, grad_rho_l, grad_u_l, 0)
            f_local_mx = self.gks_flux_local(rho_f_y, vel_l, grad_rho_l, grad_u_l, 1)
            f_local_my = self.gks_flux_local(rho_f_y, vel_l, grad_rho_l, grad_u_l, 2)
            # Convert local momentum flux [normal-momentum, tangential-momentum]
            # back to world momentum flux [x-momentum, y-momentum].
            self.face_y[i, j] = ti.Vector([f0, f_local_my, f_local_mx])

    @ti.kernel
    def update(self):
        for i, j in self.rho:
            im = self.wrap_i(i - 1)
            jm = self.wrap_j(j - 1)

            div = (self.face_x[i, j] - self.face_x[im, j]) + (self.face_y[i, j] - self.face_y[i, jm])

            rho_old = self.rho[i, j]
            mom_old = rho_old * self.u[i, j]

            rho_new = rho_old - self.dt * div[0]
            mom_new = mom_old - self.dt * ti.Vector([div[1], div[2]])

            self.rho_next[i, j] = rho_new
            self.u_next[i, j] = mom_new / rho_new

    @ti.kernel
    def swap(self):
        for i, j in self.rho:
            self.rho[i, j] = self.rho_next[i, j]
            self.u[i, j] = self.u_next[i, j]

    def step(self):
        self.compute_face_fluxes()
        self.update()
        self.swap()

    def kinetic_energy(self):
        rho_np = self.rho.to_numpy()
        u_np = self.u.to_numpy()
        return float(0.5 * np.mean(rho_np * np.sum(u_np * u_np, axis=-1)))

    def density_stats(self):
        rho_np = self.rho.to_numpy()
        return float(rho_np.min()), float(rho_np.mean()), float(rho_np.max())

    def snapshot_arrays(self):
        u = self.u.to_numpy()
        ux = u[:, :, 0]
        return ux

    def shear_wave_error(self, time):
        ux = self.snapshot_arrays()
        y = (np.arange(self.ny, dtype=np.float32) + 0.5) / float(self.ny)
        amp0 = float(self.mach) * self.cs

        # The solver uses unit grid spacing. For a wave sin(2*pi*j/ny), the
        # discrete Laplacian eigenvalue is -4*sin(pi/ny)^2.
        k_eff2 = 4.0 * math.sin(math.pi / float(self.ny)) ** 2
        amp = amp0 * math.exp(-self.nu * k_eff2 * time)
        exact_1d = amp * np.sin(2.0 * math.pi * y)
        exact_ux = np.broadcast_to(exact_1d[None, :], ux.shape)

        err_ux = ux - exact_ux
        l2 = float(np.sqrt(np.mean(err_ux * err_ux)))
        linf = float(np.max(np.abs(err_ux)))
        # amp_num = float(0.5 * (np.max(ux) - np.min(ux)))
        amp_num = 2.0 * np.mean(ux * np.sin(2.0 * math.pi * y[None, :]))
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


def main():
    ti.init(arch=ti.gpu)

    sim = SinglePhaseGKS2D()
    sim.init()
    gui = ti.GUI(
        f"Single-phase GKS",
        res=(sim.nx, sim.ny),
        fast_gui=False,
    )

    s = 0
    sim_time = 0.0
    while gui.running:
        sim.step()
        gui.set_image(sim.render_image())
        gui.show()
        s += 1
        sim_time += sim.dt
        if s % 100 == 0:
            l2, linf, amp_num, amp_exact = sim.shear_wave_error(sim_time)
            msg = f" l2={l2:.6e} linf={linf:.6e} amp_num={amp_num:.6e} amp_exact={amp_exact:.6e}"
            print(msg)


if __name__ == "__main__":
    main()
