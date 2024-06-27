import dolfin as df
from dataclasses import dataclass


@dataclass
class NavierStokesParameters:
    dt: float  # time-step
    rho: float  # density
    mu: float  # viscosity


@dataclass
class NeoHookeanParameters:
    dt: float  # time-step
    rho: float  # density
    g: float  # elastic modulus


@dataclass
class SaintVenantParameters:
    dt: float  # time-step
    rho: float  # density
    mu_s: float  # elastic modulus
    lam: float


def grad_x(func, def_grad):
    return df.grad(func) * df.inv(def_grad)


def div_x(func, def_grad):
    return df.tr(grad_x(func, def_grad))


def navier_stokes_ale(
    w: df.Function,
    w0: df.Function,
    w_: df.TestFunction,
    parameters: NavierStokesParameters,
    dx: df.Measure,
    u_init: df.Function = None
):
    (v, u, p) = df.split(w)
    (v_, u_, p_) = df.split(w_)
    (v0, u0, _) = df.split(w0)
    k = 1.0 / parameters.dt
    rho = parameters.rho
    gdim = len(v)
    identity = df.Identity(gdim)
    if u_init is not None:
        def_grad = (
            (identity + df.grad(u) * df.inv(identity + df.grad(u_init)))
            * (identity + df.grad(u_init))
        )
        det_init = df.det(identity + df.grad(u_init))
    else:
        def_grad = identity + df.grad(u)
        det_init = 1
    # deformation gradien
    cauchy_stress = 2 * parameters.mu * df.sym(grad_x(v, def_grad)) - p * identity
    determinant = df.det(def_grad)
    return (
        (rho * df.inner(k * (v - v0), v_)) * determinant * dx
        + df.inner(cauchy_stress, grad_x(v_, def_grad)) * determinant * dx
        + rho * df.inner(grad_x(v, def_grad) * (v - k * (u - u0)), v_) * determinant * dx
        + df.inner(div_x(v, def_grad), p_) * determinant * dx
    )


def saint_venant(
    w: df.Function,
    w0: df.Function,
    w_: df.TestFunction,
    parameters: SaintVenantParameters,
    dx: df.Measure,
    u_init: df.Function = None,
):
    (v, u, p) = df.split(w)
    (v_, u_, p_) = df.split(w_)
    (v0, u0, _) = df.split(w0)
    k = 1.0 / parameters.dt
    rho = parameters.rho
    gdim = len(v)
    identity = df.Identity(gdim)
    if u_init is not None:
        def_grad_init = identity + df.grad(u_init)
        det_init = df.det(def_grad_init)
    else:
        def_grad_init = identity
        det_init = 1.0 
    # deformation gradient
    def_grad = identity + df.grad(u) * df.inv(def_grad_init)
    C = (def_grad.T) * def_grad
    lam = parameters.lam
    mu_s = parameters.mu_s
    piola = def_grad * (
        0.5 * lam * df.tr(C - identity) * identity + mu_s * (C - identity)
    )
    return (
        rho * df.inner(k * (v - v0), v_) * det_init * dx
        + df.inner(piola, df.grad(v_) * df.inv(def_grad_init)) * det_init * dx
        + df.inner(k * (u - u0), u_) * det_init * dx
        - df.inner(v, u_) * det_init * dx
        + df.inner(df.grad(p), df.grad(p_)) * dx
    )


def neo_hookean(
    w: df.Function,
    w0: df.Function,
    w_: df.TestFunction,
    parameters: NeoHookeanParameters,
    dx: df.Measure,
):
    (v, u, p) = df.split(w)
    (v_, u_, p_) = df.split(w_)
    (v0, u0, _) = df.split(w0)
    k = 1.0 / parameters.dt
    rho = parameters.rho
    gdim = len(v)
    identity = df.Identity(gdim)
    # deformation gradient
    def_grad = identity + df.grad(u)
    piola = df.det(def_grad) * (-p * df.inv(def_grad).T + 2 * parameters.g * def_grad)
    C = (def_grad.T) * def_grad
    nu0 = 1000
    nu1 = 2000
    piola = def_grad * (
        0.5 * nu0 * df.tr(C - identity) * identity + nu1 * (C - identity)
    )

    determinant = df.det(def_grad)
    return (
        rho * df.inner(k * (v - v0), v_) * dx
        + df.inner(piola, df.grad(v_)) * dx
        + df.inner(k * (u - u0), u_) * dx
        - df.inner(v, u_) * dx
        + df.inner(determinant - 1, p_) * dx
    )

def navier_stokes_ale_force(w, w0, w_, mu, normal, sign_f, sign_s, theta=1.0, u_init=None):
    (v, u, p) = df.split(w)
    (v_, u_, p_) = df.split(w_)
    (v0, u0, _) = df.split(w0)
    gdim = len(v)
    I = df.Identity(gdim)
    if u_init is not None:
        F = (
            (I + df.grad(u(sign_s)) * df.inv(I + df.grad(u_init(sign_s))))
            * (I + df.grad(u_init(sign_s)))
        )
        # F = I + df.grad(u(sign_s)) * df.inv(I + df.grad(u_init(sign_s)))
        F0 = I + df.grad(u0(sign_s)) * df.inv(I + df.grad(u_init(sign_s)))
        det_init = df.det(I + df.grad(u_init(sign_s)))
    else:
        F = I + df.grad(u(sign_s))
        F0 = I + df.grad(u0(sign_s))
        det_init = 1
    T = 2 * mu * df.sym(df.dot(df.grad(v(sign_f)), df.inv(F))) - p(sign_f) * I
    T0 = 2 * mu * df.sym(df.dot(df.grad(v0(sign_f)), df.inv(F0))) - p(sign_f) * I
    if theta == 1.0:
        return (
            df.det(F) * df.inner(
                T * (df.inv(F.T)) * normal(sign_f),
                v_(sign_s)
            ) * df.dS(metadata={"quadrature_degree": 4})
        )
    return (
        theta * df.det(F) * df.inner(T * (df.inv(F.T)) * normal(sign_f), v_(sign_s)) * df.dS(metadata={"quadrature_degree": 4})
        + (1 - theta) * df.det(F0) * df.inner(T0 * (df.inv(F0.T)) * normal(sign_f), v_(sign_s)) * df.dS(metadata={"quadrature_degree": 4})
    )


def navier_slip(v, u, p, v_, u_, p_, normal_ref, theta, mu, sign_f, sign_s, gamma=3.08, u_init=None):
    gdim = len(v)
    dS = df.dS(metadata={"quadrature_degree": 8})
    I = df.Identity(gdim)
    if u_init is not None:
        F_f = (
            (I + df.grad(u(sign_f)) * df.inv(I + df.grad(u_init(sign_f))))
            * (I + df.grad(u_init(sign_f)))
        )
        F_s = (
            (I + df.grad(u(sign_s)) * df.inv(I + df.grad(u_init(sign_s))))
            * (I + df.grad(u_init(sign_s)))
        )
        # F = I + df.grad(u(sign_s)) * df.inv(I + df.grad(u_init(sign_s)))
    else:
        F_f = I + df.grad(u(sign_f))
        F_s = I + df.grad(u(sign_s))
    n_norm = df.sqrt(df.inner(df.inv(F_f.T) * normal_ref(sign_f), df.inv(F_f.T) * normal_ref(sign_f)))
    normal = df.inv(F_f.T) * normal_ref(sign_f) / n_norm
    vt_f = v(sign_f) - df.inner(v(sign_f), normal) * normal
    vt_f_ = v_(sign_f) - df.inner(v_(sign_f), normal) * normal
    vt_s = v(sign_s) - df.inner(v(sign_s), normal) * normal
    T = 2 * mu * df.sym(df.dot(df.grad(v(sign_f)), df.inv(F_s))) - p(sign_f) * I
    T_ = 2 * mu * df.sym(df.dot(df.grad(v_(sign_f)), df.inv(F_s))) - p_(sign_f) * I
    J = df.det(F_f)
    navier_slip = (
        # T(phi_v, phi_p) * (v_f - v_s)
        J * df.inner(v(sign_f) - v(sign_s), normal) * df.inner(
            T_ * df.inv(F_f.T) * normal_ref(sign_f), normal
        ) * dS
        # - J * T(v,p) * F^{-T} n *(phi_v, n)
        - J * df.inner(v_(sign_f), normal)
        * df.inner(T * df.inv(F_f.T) * normal_ref(sign_f), normal) * dS
        # penality
        + J * df.inner(v(sign_f) - v(sign_s), normal)
        * 200000 * df.inner(v_(sign_f), normal) * dS
    )
    # slip contribution
    if theta != 0:
        navier_slip += J * n_norm * (theta / (gamma * (1.0 - theta))) * df.inner(
            vt_f - vt_s, vt_f_
        ) * dS
    return navier_slip