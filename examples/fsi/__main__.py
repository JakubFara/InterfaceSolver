import dolfin as df
from mpi4py import MPI
from InterfaceSolver import NonlinearInterfaceSolver
from InterfaceSolver import interface
from dataclasses import dataclass
from petsc4py.PETSc import Sys
from ufl import atan_2

from forms import (
    navier_stokes_ale,
    NavierStokesParameters,
    NeoHookeanParameters,
    SaintVenantParameters,
    neo_hookean,
    saint_venant,
    navier_stokes_ale_force,
    navier_slip,
)

theta_scheme_partam = 1.0
theta = 0.5 
# maximal_radius = 0.016
maximal_radius = 0.018

@dataclass
class Parameters:
    rho_s: float = 1.0e3
    nu_s: float = 0.49
    mu_s: float = 1e4
    rho_f: float = 1.05e3
    mu_f: float = 3.8955e-3
    # mu_f: float = 1.0 
    E: float = 2 * mu_s * (1 + nu_s)
    lam: float = E * nu_s / ((1 + nu_s) * (1 - 2 * nu_s))


parameters = Parameters()


@dataclass
class Lables:
    solid: int = 1
    fluid: int = 2
    solid_sign: str = "+"
    fluid_sign: str = "-"

labels = Lables()


dt = 0.005
t = 0.0
t_end = 4.0

comm = MPI.COMM_WORLD
size = comm.Get_size()

df.parameters["std_out_all_processes"] = False
with df.HDF5File(comm, "data/tube3d_discontinuous.h5", "r") as h5_file:
    mesh = df.Mesh(comm)
    h5_file.read(mesh, "/mesh", False)
    marker = df.MeshFunction('size_t', mesh, 3)
    h5_file.read(marker, "/cell_marker")

directory = f"results/tube3d/theta{theta}/radius{maximal_radius}/"

radius = 0.012
radius2 = 0.002
tube_length = 0.044

u_init_expr = df.Expression(
    (
        "0.005 * x[0] / max(pow(pow(x[0], 2) + pow(x[1], 2), 0.5), 0.000001) * min(pow(pow(x[0], 2) + pow(x[1], 2), 0.5), radius) / radius * exp(-10000 * pow(x[2], 2)) ",
        "0.005 * x[1] / max(pow(pow(x[0], 2) + pow(x[1], 2), 0.5), 0.000001) * min(pow(pow(x[0], 2) + pow(x[1], 2), 0.5), radius) / radius * exp(-10000 * pow(x[2], 2)) ",
        "0",
    ),
    degree=3,
    radius=radius,
    length=tube_length,
)

class UInit(df.UserExpression):
    def __init__(self, r_max, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r_max = r_max
        self.set_parameters()

    def eval(self, value, x):
        exp_shape = self.exp_shape(x)
        phi = atan_2(-x[1], -x[0])
        if x[2] < 0:
            D = self.D_in
        else:
            D = self.D_out
        # r_scale = min(df.sqrt(x[0]**2 + x[1]**2) / (self.D_in / 2), 1) 
        r_scale = df.sqrt(x[0]**2 + x[1]**2) / (self.D_in / 2) 
        value[0] = (
            r_scale * (D / 2 + (self.R + self.r - D / 2) * exp_shape) * df.cos(phi)
            - r_scale * self.lam * self.r * df.exp(-8000 * x[2]**2) * df.cos(((self.R + self.r) / self.r) * phi)
            - x[0]
        )
        value[1] = (
            r_scale * (D / 2 + (self.R + self.r - D / 2) * exp_shape) * df.sin(phi)
            - r_scale * self.lam * self.r * df.exp(-8000 * x[2]**2) * df.sin(((self.R + self.r) / self.r) * phi)
            - x[1]
        )
        value[2] = 0 
    
    def exp_shape(self, x):
        if x[2] < 0:
            return df.exp(- self.popt_in[0] * x[2]**2 - self.popt_in[1] * x[2]**4)
        else:
            return df.exp(- self.popt_out[0] * x[2]**2 - self.popt_out[1] * x[2]**4)

    def set_parameters(self):
        self.L = 0.044     
        self.lam = 0.5 
        self.D_in = 0.024 
        self.D_out = 0.026 
        if self.r_max == 0.016:
           self.r = 0.00355
           self.R = 0.01065
           self.popt_in = [6.29230105e+03, 7.24254548e+07]
           self.popt_out = [2.13704667e+03, 1.00960705e+08]
        elif self.r_max == 0.018:
           self.r = 0.004
           self.R = 0.012
           self.popt_in = [5.66912918e+03, 7.66378929e+07]
           self.popt_out = [3.04617628e+03, 9.46300951e+07]
        elif self.r_max == 0.02:
           self.r = 0.00445 
           self.R = 0.01335
           self.popt_in = [5.36141778e+03, 7.87273868e+07]
           self.popt_out = [3.44279531e+03, 9.18833710e+07]

    def value_shape(self):
        return (3, )

u_init_expr = UInit(maximal_radius)

labels.fluid_sign = "-"  # plus corresponds to the cell val
labels.solid_sign = "+"
cell_val = labels.solid  # bottom

# function spaces
Ep = df.FiniteElement("CG", mesh.ufl_cell(), 1)
Eu = df.VectorElement("CG", mesh.ufl_cell(), 2)
Ev = df.VectorElement("CG", mesh.ufl_cell(), 2)
En = df.VectorElement("DG", mesh.ufl_cell(), 1)

V = df.FunctionSpace(mesh, df.MixedElement([Ev, Eu, Ep]))
V_u = df.FunctionSpace(mesh, Eu)
V_n = df.FunctionSpace(mesh, En)
w = df.Function(V)
w0 = df.Function(V)
w_ = df.TestFunction(V)
(v, u, p) = df.split(w)
(v0, u0, p0) = df.split(w0)
(v_, u_, p_) = df.split(w_)
# u_init = df.Function(V_u)
u_init = df.project(u_init_expr, V_u)

with df.XDMFFile(comm, f"{directory}/u_init.xdmf") as xdmf_u_init:
    xdmf_u_init.parameters["flush_output"] = True
    xdmf_u_init.parameters["functions_share_mesh"] = True
    u_init.rename("u_init", "u_init")
    xdmf_u_init.write(u_init, 0)

# Boundary
labels_bndry = {
    "inflow_f": 1,
    "outflow_f": 2,
    "inflow_s": 3,
    "outflow_s": 4,
    "mid_s": 5,
    "mid_f": 6,
}

bndry_marker = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
# interface = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
for facet in df.facets(mesh):
    cells = [c for c in df.cells(facet)]
    mid_facet = facet.midpoint()
    x = float(mid_facet.x())
    y = float(mid_facet.y())
    z = float(mid_facet.z())
    r = x**2 + y**2
    eps = 1e-6
    if r < radius**2 and abs(z + tube_length/2) < eps:
        bndry_marker[facet] = labels_bndry["inflow_f"]
    elif r > radius**2 and abs(z + tube_length/2) < eps:
        bndry_marker[facet] = labels_bndry["inflow_s"]
    elif r < radius**2 and abs(z - tube_length/2) < eps:
        bndry_marker[facet] = labels_bndry["outflow_f"]
    elif r > radius**2 and abs(z - tube_length/2) < eps:
        bndry_marker[facet] = labels_bndry["outflow_s"]
    elif radius**2 + 0.000003 > r > radius**2 - 0.000003:
        cell = [c for c in df.cells(facet)][0]
        if marker[cell] == labels.fluid:
            bndry_marker[facet] = labels_bndry["mid_f"]
        else:
            bndry_marker[facet] = labels_bndry["mid_s"]

# save markers for visualizarion
# file_xml = df.File("bndry_marker.pvd")
# file_xml << bndry_marker

def inflow_average(t):
    T = 1.0
    x0 = 0.3
    V = 0.7
    tt = t % T
    if(tt < x0):
        return -4.0 * V / (x0*x0) * tt * (tt-x0)
    else:
        return 0.0

def velocity(t: float) -> float:
    tp = t%1.0
    if tp > 0.3:
        return 0.0
    else:
        return - 0.7 * tp * (tp - 0.3) / (0.15 * (0.15))

vel_avg = 0.65

if theta == 1:
    profile = "v*2.0*(pow(r, 2) - (pow(x[0], 2) + pow(x[1], 2)))/(r*r)"
    inflow_expr = df.Expression(
        (0, 0, profile), r=radius, v=vel_avg, degree=2
    )
else:
    profile = (
        "(v * (4.0 * gamma * mu * (1.0 - theta) * r +"
        "2.0 * theta * (pow(r, 2) - pow(x[0], 2) - pow(x[1], 2) )))"
        "/ ( 4.0 * gamma * mu * (1 - theta) * r + theta * r * r)"
    )
    inflow_expr = df.Expression(
        (0, 0, profile),
        r=radius,
        gamma=3.08,
        mu=parameters.mu_f,
        theta=theta,
        # t=0.9,
        v=vel_avg,
        degree=4
    )


zero_vec = df.Constant((0.0, 0.0, 0.0))
bc_v_fluid_inflow = df.DirichletBC(V.sub(0), inflow_expr, bndry_marker, labels_bndry["inflow_f"])
bc_v_fluid_outflow = df.DirichletBC(V.sub(0), zero_vec, bndry_marker, labels_bndry["outflow_f"])

bc_v_solid_inflow = df.DirichletBC(V.sub(0), zero_vec, bndry_marker, labels_bndry["inflow_s"])
bc_v_solid_outflow = df.DirichletBC(V.sub(0), zero_vec, bndry_marker, labels_bndry["outflow_s"])

bc_u_fluid_inflow = df.DirichletBC(V.sub(1), zero_vec, bndry_marker, labels_bndry["inflow_f"])
bc_u_fluid_outflow = df.DirichletBC(V.sub(1), zero_vec, bndry_marker, labels_bndry["outflow_f"])

bc_u_solid_inflow = df.DirichletBC(V.sub(1), zero_vec, bndry_marker, labels_bndry["inflow_s"])
bc_u_solid_outflow = df.DirichletBC(V.sub(1), zero_vec, bndry_marker, labels_bndry["outflow_s"])

bcs_solid = [
    bc_v_solid_outflow,
    bc_v_solid_inflow,
    bc_u_solid_outflow,
    bc_u_solid_inflow,
]

bcs_fluid = [
    bc_v_fluid_inflow,
    bc_u_fluid_outflow,
    bc_u_fluid_inflow,
]

# bc_out = DirichletBC(V.sub(0), zero_vec, outflow_inner)
bcm_v = df.DirichletBC(V.sub(0), zero_vec, bndry_marker, labels_bndry["mid_f"])
bcm_u = df.DirichletBC(V.sub(1), zero_vec, bndry_marker, labels_bndry["mid_f"])

with df.XDMFFile(comm, f"{directory}/v.xdmf") as xdmf_v:
    xdmf_v.parameters["flush_output"] = True
    xdmf_v.parameters["functions_share_mesh"] = True
    xdmf_file_v = xdmf_v

with df.XDMFFile(comm, f"{directory}/u.xdmf") as xdmf_u:
    xdmf_u.parameters["flush_output"] = True
    xdmf_u.parameters["functions_share_mesh"] = True
    xdmf_file_u = xdmf_u


def interface_func(x, y, z):
    return x**2 + y**2 - radius**2


interface_label = interface(mesh, interface_func, val=1, eps=0.000003)
dX = df.Measure("dx")(domain=mesh, subdomain_data=marker, metadata={"quadrature_degree": 8})
ds = df.Measure("ds")(domain=mesh, subdomain_data=bndry_marker, metadata={"quadrature_degree": 8})

# ufl forms
# normal vector
normal = df.Expression(
    (
        "x[0]/pow(pow(x[0], 2) + pow(x[1], 2), 0.5)",
        "x[1]/pow(pow(x[0], 2) + pow(x[1], 2), 0.5)",
        "0.0",
    ),
    degree=4,
)
# normal = df.FacetNormal(mesh)
options = {
    "snes_": {
        "rtol": 1.0e-8,
        "atol": 1.0e-8,
        "stol": 1.0e-8,
        "max_it": 4,
        "type": "ksponly",
        # 'type': 'newtonls',
        # 'linesearch_type': "basic",
        # 'linesearch_type': "bt",
        # 'linesearch_damping': 0.999
    },
    "pc_": {"type": "lu", "factor_mat_solver_type": "mumps"},
    "mat_": {
        "type": "aij",
        "mumps_": {
            # 'icntl_1': 1,
            "cntl_1": 1e-5,
            "icntl_14": 200,
            # 'icntl_24': 1
        },
    },
    "ksp_": {
        "type": "preonly",
        "rtol": 1e-5,
        # 'atol': 1e-2
    },
}

solver = NonlinearInterfaceSolver(
    w, marker, interface_label, interface_value=1, cell_val=cell_val, params=options
)

parameters_fluid = NavierStokesParameters(mu=parameters.mu_f, rho=parameters.rho_f, dt=dt)
# parameters_solid = NeoHookeanParameters(g=mu1, rho=parameters.rho_s, dt=dt)
parameters_solid = SaintVenantParameters(
    mu_s=parameters.mu_s, lam=parameters.lam, rho=parameters.rho_s, dt=dt
)

def negpart(s):
    return df.conditional(df.gt(s, 0.0), 0.0, 1.0)*s

out_penal = (
    - 0.5 * parameters_fluid.rho
    * negpart(df.inner(v, df.Constant((0.0, 0.0, 1.0))))
    * df.inner(v, v_) * ds(labels_bndry["outflow_f"])
)

a_fluid = (
    navier_stokes_ale(w, w0, w_, parameters_fluid, dX(labels.fluid), u_init=u_init)
    + df.inner(df.grad(u), df.grad(u_)) * dX(labels.fluid)
    + out_penal
)

# a1 = neo_hookean(w, w0, w_, parameters_solid, dX(labels.solid))
a_solid = saint_venant(w, w0, w_, parameters_solid, dX(labels.solid), u_init=u_init)

a_interface = (
    navier_stokes_ale_force(
        w, w0, w_,
        parameters_fluid.mu, normal,
        labels.fluid_sign, labels.solid_sign,
        theta=1.0, u_init=u_init
    )
)

dirichlet_interface = [
    ((1, 0), None, labels.fluid_sign),  # u_x
    ((1, 1), None, labels.fluid_sign),  # u_y
    ((1, 2), None, labels.fluid_sign),  # u_z
]

if theta == 1:
    dirichlet_interface += [
        ((0, 0), None, labels.fluid_sign),  # v_x
        ((0, 1), None, labels.fluid_sign),  # v_y
        ((0, 2), None, labels.fluid_sign),  # v_z
    ]
    bcs_zero = [bcm_u, bcm_v]
else:
    a_interface += navier_slip(
        v, u, p, v_, u_, p_,
        normal, theta, parameters_fluid.mu,
        labels.fluid_sign, labels.solid_sign,
        gamma=3.08, u_init=u_init
    )
    bcs_zero = [bcm_u]

# a0 += l0
t += dt
# inflow_expr.t = t
while t < t_end:
    Sys.Print(f"    t = {t}")
    inflow_expr.v = velocity(t)
    # solve
    for i in range(5):
        res, conv_reason = solver.solve(
            a_solid,
            a_fluid,
            a_interface,
            bcs0=bcs_solid,
            bcs1=bcs_fluid,
            bcs_zero0=[],
            bcs_zero1=bcs_zero,
            dirichlet_interface=dirichlet_interface,
        )
        df.info(f"{res}")
        if res.get(1) == None:
            continue
        if res[1] < 1e-8:
            break
    w0.assign(w)
    (v, u, p) = w.split(True)
    # save and plot
    u.rename("u", "u")
    v.rename("v", "v")
    xdmf_v.write(v, t)
    xdmf_u.write(u, t)
    xdmf_u_init.write(u_init, t)
    t += dt
