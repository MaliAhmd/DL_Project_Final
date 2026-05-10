import taichi as ti
import math
import numpy as np

real = ti.f32

@ti.data_oriented
class DiffMPMModel:
    def __init__(self, dim=2, n_grid=128, max_steps=2048, dt=1e-3):
        self.dim = dim
        self.n_grid = n_grid
        self.dx = 1 / n_grid
        self.inv_dx = 1 / self.dx
        self.dt = dt
        self.p_vol = 1
        self.E = 10
        self.mu = self.E
        self.la = self.E
        self.max_steps = max_steps
        self.gravity = 3.8
        
        self.n_particles = 0
        self.n_solid_particles = 0
        self.n_actuators = 0
        self.n_sin_waves = 4
        
        self.scalar = lambda: ti.field(dtype=real)
        self.vec = lambda: ti.Vector.field(self.dim, dtype=real)
        self.mat = lambda: ti.Matrix.field(self.dim, self.dim, dtype=real)

        # Fields (will be allocated later)
        self.actuator_id = ti.field(ti.i32)
        self.particle_type = ti.field(ti.i32)
        self.x, self.v = self.vec(), self.vec()
        self.grid_v_in, self.grid_m_in = self.vec(), self.scalar()
        self.grid_v_out = self.vec()
        self.C, self.F = self.mat(), self.mat()
        self.loss = self.scalar()
        self.weights = self.scalar()
        self.bias = self.scalar()
        self.x_avg = self.vec()
        self.actuation = self.scalar()
        
        self.actuation_omega = 20
        self.act_strength = 4

    def allocate_fields(self, n_particles, n_actuators):
        self.n_particles = n_particles
        self.n_actuators = n_actuators
        
        ti.root.dense(ti.ij, (self.n_actuators, self.n_sin_waves)).place(self.weights)
        ti.root.dense(ti.i, self.n_actuators).place(self.bias)
        ti.root.dense(ti.ij, (self.max_steps, self.n_actuators)).place(self.actuation)
        ti.root.dense(ti.i, self.n_particles).place(self.actuator_id, self.particle_type)
        ti.root.dense(ti.k, self.max_steps).dense(ti.l, self.n_particles).place(self.x, self.v, self.C, self.F)
        ti.root.dense(ti.ij, self.n_grid).place(self.grid_v_in, self.grid_m_in, self.grid_v_out)
        ti.root.place(self.loss, self.x_avg)
        ti.root.lazy_grad()

    @ti.kernel
    def clear_grid(self):
        for i, j in self.grid_m_in:
            self.grid_v_in[i, j] = [0, 0]
            self.grid_m_in[i, j] = 0
            self.grid_v_in.grad[i, j] = [0, 0]
            self.grid_m_in.grad[i, j] = 0
            self.grid_v_out.grad[i, j] = [0, 0]

    @ti.kernel
    def p2g(self, f: ti.i32):
        for p in range(self.n_particles):
            base = ti.cast(self.x[f, p] * self.inv_dx - 0.5, ti.i32)
            fx = self.x[f, p] * self.inv_dx - ti.cast(base, ti.i32)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            new_F = (ti.Matrix.diag(dim=2, val=1) + self.dt * self.C[f, p]) @ self.F[f, p]
            J = (new_F).determinant()
            if self.particle_type[p] == 0:  # fluid
                sqrtJ = ti.sqrt(J)
                new_F = ti.Matrix([[sqrtJ, 0], [0, sqrtJ]])

            self.F[f + 1, p] = new_F
            r, s = ti.polar_decompose(new_F)
            act_id = self.actuator_id[p]
            act = self.actuation[f, ti.max(0, act_id)] * self.act_strength
            if act_id == -1:
                act = 0.0

            A = ti.Matrix([[0.0, 0.0], [0.0, 1.0]]) * act
            cauchy = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
            mass = 0.0
            if self.particle_type[p] == 0:
                mass = 4
                cauchy = ti.Matrix([[1.0, 0.0], [0.0, 0.1]]) * (J - 1) * self.E
            else:
                mass = 1
                cauchy = 2 * self.mu * (new_F - r) @ new_F.transpose() + \
                         ti.Matrix.diag(2, self.la * (J - 1) * J)
            cauchy += new_F @ A @ new_F.transpose()
            stress = -(self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * cauchy
            affine = stress + mass * self.C[f, p]
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    offset = ti.Vector([i, j])
                    dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * self.dx
                    weight = w[i][0] * w[j][1]
                    self.grid_v_in[base + offset] += weight * (mass * self.v[f, p] + affine @ dpos)
                    self.grid_m_in[base + offset] += weight * mass

    @ti.kernel
    def grid_op(self, bound: ti.i32, coeff: ti.f32):
        for i, j in self.grid_m_in:
            inv_m = 1 / (self.grid_m_in[i, j] + 1e-10)
            v_out = inv_m * self.grid_v_in[i, j]
            v_out[1] -= self.dt * self.gravity
            if i < bound and v_out[0] < 0:
                v_out[0] = 0
                v_out[1] = 0
            if i > self.n_grid - bound and v_out[0] > 0:
                v_out[0] = 0
                v_out[1] = 0
            if j < bound and v_out[1] < 0:
                v_out[0] = 0
                v_out[1] = 0
                normal = ti.Vector([0.0, 1.0])
                lsq = (normal**2).sum()
                if lsq > 0.5:
                    if coeff < 0:
                        v_out[0] = 0
                        v_out[1] = 0
                    else:
                        lin = v_out.dot(normal)
                        if lin < 0:
                            vit = v_out - lin * normal
                            lit = vit.norm() + 1e-10
                            if lit + coeff * lin <= 0:
                                v_out[0] = 0
                                v_out[1] = 0
                            else:
                                v_out = (1 + coeff * lin / lit) * vit
            if j > self.n_grid - bound and v_out[1] > 0:
                v_out[0] = 0
                v_out[1] = 0
            self.grid_v_out[i, j] = v_out

    @ti.kernel
    def g2p(self, f: ti.i32):
        for p in range(self.n_particles):
            base = ti.cast(self.x[f, p] * self.inv_dx - 0.5, ti.i32)
            fx = self.x[f, p] * self.inv_dx - ti.cast(base, real)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
            new_v = ti.Vector([0.0, 0.0])
            new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    dpos = ti.cast(ti.Vector([i, j]), real) - fx
                    g_v = self.grid_v_out[base[0] + i, base[1] + j]
                    weight = w[i][0] * w[j][1]
                    new_v += weight * g_v
                    new_C += 4 * weight * g_v.outer_product(dpos) * self.inv_dx

            self.v[f + 1, p] = new_v
            self.x[f + 1, p] = self.x[f, p] + self.dt * self.v[f + 1, p]
            self.C[f + 1, p] = new_C

    @ti.kernel
    def compute_actuation(self, t: ti.i32):
        for i in range(self.n_actuators):
            act = 0.0
            for j in ti.static(range(self.n_sin_waves)):
                act += self.weights[i, j] * ti.sin(self.actuation_omega * t * self.dt +
                                              2 * math.pi / self.n_sin_waves * j)
            act += self.bias[i]
            self.actuation[t, i] = ti.tanh(act)

    @ti.kernel
    def compute_x_avg(self, steps: ti.i32):
        for i in range(self.n_particles):
            contrib = 0.0
            if self.particle_type[i] == 1:
                contrib = 1.0 / self.n_solid_particles
            ti.atomic_add(self.x_avg[None], contrib * self.x[steps - 1, i])

    @ti.kernel
    def compute_loss(self):
        dist = self.x_avg[None][0]
        self.loss[None] = -dist
