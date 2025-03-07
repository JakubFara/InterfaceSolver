import dolfin as df
from mpi4py import MPI

from InterfaceSolver import interface
from InterfaceSolver import NonlinearInterfaceSolver
from dataclasses import dataclass
from petsc4py.PETSc import Sys
from petsc4py import PETSc

from ufl import atan_2
# from init_displacement import u_init, u_init_symmetric
import argparse
from mesh_partitioning import get_partitioned_mesh

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


df.parameters["std_out_all_processes"] = False

parser = argparse.ArgumentParser()
df.parameters["ghost_mode"] = "none"
df.parameters['form_compiler']['quadrature_degree'] = 8

parser.add_argument(
    "--shear_modulus" , "-sm",
    help="shear modulus",
    default=1e4,
    type=float,
)

parser.add_argument(
    "--radius" , "-r",
    help="radius 16 or 20 mm",
    default=0,
    type=float,
)

parser.add_argument(
    "--theta" , "-t",
    help=(
        "navire slip parameter",
    ),
    default=1.0,
    type=float,
)

parser.add_argument(
    "--refined" , "-ref",
    help=(
        "1 -- True, 0 -- False",
    ),
    default=1,
    type=int,
)

parser.add_argument(
    "--mesh_level" , "-ml",
    help=(
        "mesh_level",
    ),
    default=1,
    type=int,
)

parser.add_argument(
    "--dt" , "-dt",
    help=(
        "time-step",
    ),
    default=0.01,
    type=float,
)

args = vars(parser.parse_args())
maximal_radius = args["radius"]
theta = args["theta"]
mesh_level = args["mesh_level"]
refined = args["refined"]
dt_base = args["dt"]

radius = 0.012
radius2 = 0.002
tube_length = 0.044
dt = df.Constant(dt_base)
t = 0
t_end = 3.0

theta_scheme_partam = 1.0
# maximal_radius = 0.016

load_from_checkpoint = False
@dataclass
class Parameters:
    rho_s: float = 1.0e3
    nu_s: float = 0.49
    mu_s: float = args["shear_modulus"]
    rho_f: float = 1.05e3
    mu_f: float = 3.8955e-3
    # mu_f: float = 1.0
    E: float = 2 * mu_s * (1 + nu_s)
    lam: float = E * nu_s / ((1 + nu_s) * (1 - 2 * nu_s))


parameters = Parameters()

comm = df.MPI.comm_world
rank = comm.Get_rank()
# with df.HDF5File(comm, f"data/aorta_discontinuous_lev{mesh_level}.h5", "r") as h5_file:
if refined == 0:
    mesh = get_partitioned_mesh(
        f'data/mesh_lev{mesh_level}_r16/mesh_discontinuous.h5',
        f'data/mesh_lev{mesh_level}_r16/mesh_continuous.h5'
    )
    with df.HDF5File(comm, f'data/mesh_lev{mesh_level}_r16/mesh_discontinuous.h5', "r") as h5_file:
        # first we need to create an empty mesh
        # mesh = df.Mesh(comm)
        # load the data stored in `/mesh` to the mesh
        # h5_file.read(mesh, "/mesh", False)
        mesh2 = df.Mesh(mesh)
        # the dimension of the mesh
        # dim = mesh.geometry().dim()
        dim = 3
        # we need to create the empty meshfunctions at first
        marker = df.MeshFunction('size_t', mesh, dim)
        bndry_marker = df.MeshFunction('size_t', mesh, dim - 1)
        interface_marker = df.MeshFunction('size_t', mesh, dim - 1)
        # we load the data to the subdomains markers
        h5_file.read(marker, "/cell_marker")
        h5_file.read(bndry_marker, "/facet_marker")
        h5_file.read(interface_marker, "/interface_marker")
else:
    mesh = get_partitioned_mesh(
        f'data/mesh_lev{mesh_level}_r16_refined/mesh_discontinuous.h5',
        f'data/mesh_lev{mesh_level}_r16_refined/mesh_continuous.h5'
    )
    with df.HDF5File(comm, f'data/mesh_lev{mesh_level}_r16_refined/mesh_discontinuous.h5', "r") as h5_file:
        # first we need to create an empty mesh
        # mesh = df.Mesh(comm)
        # load the data stored in `/mesh` to the mesh
        # h5_file.read(mesh, "/mesh", False)
        mesh2 = df.Mesh(mesh)
        # the dimension of the mesh
        # dim = mesh.geometry().dim()
        dim = 3
        # we need to create the empty meshfunctions at first
        marker = df.MeshFunction('size_t', mesh, dim)
        bndry_marker = df.MeshFunction('size_t', mesh, dim - 1)
        interface_marker = df.MeshFunction('size_t', mesh, dim - 1)
        # we load the data to the subdomains markers
        h5_file.read(marker, "/cell_marker")
        h5_file.read(bndry_marker, "/facet_marker")
        h5_file.read(interface_marker, "/interface_marker")


directory = f"./results/theta{theta}/radius{maximal_radius}/g{parameters.mu_s}/ml{mesh_level}/"
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
u_init = None

df.info(f"{V.dim()=}")

labels = {
      "solid": 1,
      "fluid": 2,
      "inflow_s": 1,
      "inflow_f": 2,
      "outflow_s": 3,
      "outflow_f": 4,
      "out": 5,
      "interface": 6,
      "mid_f": 7,
      "mid_s": 8,
      "solid_sign": "+",
      "fluid_sign": "-",
}
cell_val = labels["solid"]


if u_init != None:
    with df.XDMFFile(comm, f"{directory}/u_init.xdmf") as xdmf_u_init:
        xdmf_u_init.parameters["flush_output"] = True
        xdmf_u_init.parameters["functions_share_mesh"] = True
        u_init.rename("u_init", "u_init")
        xdmf_u_init.write(u_init, 0)
 
file_xml = df.File("bndry_marker.pvd")
file_xml << bndry_marker

file_xml = df.File(f"{directory}/marker.pvd")
file_xml << marker

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
    # profile = "v*2.0*(pow(r, 2) - (pow(x[0], 2) + pow(x[1], 2)))/(r*r)"
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
bc_v_fluid_inflow = df.DirichletBC(V.sub(0), inflow_expr, bndry_marker, labels["inflow_f"])
bc_v_fluid_outflow = df.DirichletBC(V.sub(0), zero_vec, bndry_marker, labels["outflow_f"])

bc_v_solid_inflow = df.DirichletBC(V.sub(0), zero_vec, bndry_marker, labels["inflow_s"])
bc_v_solid_outflow = df.DirichletBC(V.sub(0), zero_vec, bndry_marker, labels["outflow_s"])

bc_u_fluid_inflow = df.DirichletBC(V.sub(1), zero_vec, bndry_marker, labels["inflow_f"])
bc_u_fluid_outflow = df.DirichletBC(V.sub(1), zero_vec, bndry_marker, labels["outflow_f"])

bc_u_solid_inflow = df.DirichletBC(V.sub(1), zero_vec, bndry_marker, labels["inflow_s"])
bc_u_solid_outflow = df.DirichletBC(V.sub(1), zero_vec, bndry_marker, labels["outflow_s"])

# bc_v_cylinder = DirichletBC(V.sub(0), zero_vec, cylinder)
# bc_u_cylinder = DirichletBC(V.sub(1), zero_vec, cylinder)
bcs_solid = [
    bc_v_solid_outflow,
    bc_v_solid_inflow,
    bc_u_solid_outflow,
    bc_u_solid_inflow,
    # bc_v_cylinder,
    # bc_u_cylinder,
]

bcs_fluid = [
    # bc_v_fluid_outflow,
    bc_v_fluid_inflow,
    bc_u_fluid_outflow,
    bc_u_fluid_inflow,
]

bcm_v = df.DirichletBC(V.sub(0), zero_vec, bndry_marker, labels["mid_f"])
bcm_u = df.DirichletBC(V.sub(1), zero_vec, bndry_marker, labels["mid_f"])

with df.XDMFFile(comm, f"{directory}/v.xdmf") as xdmf_v:
    xdmf_v.parameters["flush_output"] = True
    xdmf_v.parameters["functions_share_mesh"] = True
    xdmf_file_v = xdmf_v

with df.XDMFFile(comm, f"{directory}/u.xdmf") as xdmf_u:
    xdmf_u.parameters["flush_output"] = True
    xdmf_u.parameters["functions_share_mesh"] = True
    xdmf_file_u = xdmf_u

with df.XDMFFile(comm, f"{directory}/p.xdmf") as xdmf_p:
    xdmf_p.parameters["flush_output"] = True
    xdmf_p.parameters["functions_share_mesh"] = True
    xdmf_file_p = xdmf_u


# interface_label = interface(mesh, interface_func, val=labels["interface"], eps=0.000003)
dX = df.Measure("dx")(domain=mesh, subdomain_data=marker, metadata={"quadrature_degree": 8})
ds = df.Measure("ds")(domain=mesh, subdomain_data=bndry_marker, metadata={"quadrature_degree": 8})
dxf = dX(labels["fluid"])
dxs = dX(labels["solid"])

normal = df.FacetNormal(mesh)

# options for snes-ngmres-newton-mumps with lagedd jacobian
options_snesnpc = {
    'assembly_monitor': '',
    "snes_": {
        "rtol": 1.0e-8,
        "atol": 1.0e-8,
        "stol": 1.0e-8,
        "max_it": 200,
        "type": "ngmres",
        #'ngmres_monitor': '',
        'ngmres_restart_fm_rise': '',
        'linesearch_type': 'bt',
        "monitor": '',
        'converged_reason': '',
        'max_linear_solve_fail': -1,
        'max_fail': -1,
        #"view": "",
    },
    "npc_": {
        'snes_': {
            'monitor': '',
            #'converged_reason': '',
            'type': 'newtonls',
            'linesearch_type': 'basic',
            'linesearch_damping': 0.8,
            'rtol': 0.99,   ### !!!!!!! this decides when to rebuild jacob
            #'divergence_tolerance': 1.0,
            'max_it': 1,
            "lag_jacobian": -2,
            'lag_jacobian_persists': 1,
            #'convergence_test': 'skip',
            #'norm_schedule': 0,
            'force_iteration': '',
            'max_linear_solve_fail': -1,
            'max_fail': -1,
        },
        "ksp_": {
            "type": "preonly",
            #"monitor": "",
            #"converged_reason": '',
            #"rtol": 1e-20,
            #'atol': 1e-12,
        },
        "pc_": {
            "type": "lu",
            "factor_mat_solver_type": "mumps"
        },
    },
    "mat_": {
        "type": "aij",
        "mumps_": {
            # 'icntl_1': 1,
            "cntl_1": 1e-14,
            "icntl_14": 200,
            'icntl_24': 1
        },
    },
}

# options for normal snes-newtonls-mumps with possibly lagged jacobian
options_snes = {
    'assembly_monitor': '',
    "snes_": {
        "rtol": 1.0e-8,
        "atol": 1.0e-8,
        "stol": 1.0e-8,
        "max_it": 200,
        "type": "newtonls",
        'linesearch_type': 'nleqerr',
        "monitor": '',
        'converged_reason': '',
        #"view": "",
        "lag_jacobian": -2,
        'lag_jacobian_persists': 1,
    },
    "ksp_": {
        "type": "preonly",
    },
    "pc_": {
        "type": "lu",
        "factor_mat_solver_type": "mumps"
    },
    "mat_": {
        "type": "aij",
        "mumps_": {
            # 'icntl_1': 1,
            "cntl_1": 1e-14,
            "icntl_14": 200,
            'icntl_24': 1
        },
    },
}

def my_monitor(snes, its, rnorm, *args, **kwargs):
    #Sys.Print(f"{snes.getOptionsPrefix()=} {its=} {rnorm=} {snes.reason=}")
    #Sys.Print(f"{snes.getConvergenceHistory()=}")
    if snes.hasNPC() :
        #Sys.Print(f"{snes.npc.getOptionsPrefix()=} {its=} {rnorm=} {snes.npc.reason=}")
        #Sys.Print(f"{snes.npc.getConvergenceHistory()=}")
        if snes.npc.reason<0:
            Sys.Print(f"{snes.npc.getOptionsPrefix()=} {snes.npc.reason=} do jacobian reset next.....")
            PETSc.Options().setValue('npc_snes_lag_jacobian', -2)
            snes.setFromOptions()
    return
    
solver = NonlinearInterfaceSolver(
    w, marker, interface_marker,
    interface_value=labels["interface"], cell_val=cell_val, params=options_snesnpc,
    monitor=my_monitor)


#PETSc.Options().view()

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
    * df.inner(v, v_) * ds(labels["outflow_f"])
    # + df.inner(p_out, v_) * ds(labels["outflow_f"])
)

a_fluid = (
    navier_stokes_ale(w, w0, w_, parameters_fluid, dxf, u_init=u_init)
    + df.inner(df.grad(u), df.grad(u_)) * dxf
    + out_penal
)

# a1 = neo_hookean(w, w0, w_, parameters_solid, dX(labels.solid))
a_solid = saint_venant(w, w0, w_, parameters_solid, dxs, u_init=u_init)

a_interface = (
    navier_stokes_ale_force(
        w, w0, w_,
        parameters_fluid.mu, normal,
        labels["fluid_sign"],
        labels["solid_sign"],
        theta=1.0, u_init=u_init
    )
)

dirichlet_interface = [
    ((1, 0), None, labels["fluid_sign"]),  # u_x
    ((1, 1), None, labels["fluid_sign"]),  # u_y
    ((1, 2), None, labels["fluid_sign"]),  # u_z
]

if theta == 1:
    dirichlet_interface += [
        ((0, 0), None, labels["fluid_sign"]),  # v_x
        ((0, 1), None, labels["fluid_sign"]),  # v_y
        ((0, 2), None, labels["fluid_sign"]),  # v_z
    ]
    bcs_zero = [bcm_u, bcm_v]
else:
    a_interface += navier_slip(
        v, u, p, v_, u_, p_,
        normal, theta, parameters_fluid.mu,
        labels["fluid_sign"], labels["solid_sign"],
        gamma=3.08, u_init=u_init
    )
    bcs_zero = [bcm_u]

if not load_from_checkpoint:
    with df.HDF5File(comm, directory + "result.h5", 'w') as hdf_file:
        hdf_file.write(mesh2, '/mesh')
    n = 0
else:
    with df.HDF5File(comm, directory + "result.h5", 'r') as hdf_file:
        hdf_file.read(v, f"{n}/v")
        hdf_file.read(u, f"{n}/u")
        hdf_file.read(p, f"{n}/p")
        if u_init != None:
            hdf_file.read(u_init, f"{n}/u_init")
    with open(directory + "checkpoint.txt", 'r') as ch_file:
        line = ch_file.readline().split(" ")
        n = int(line[0])
        t = float(line[1])



# a0 += l0
t += float(dt)
# inflow_expr.t = t

solver.setup(
    a_solid,
    a_fluid,
    a_interface,
    bcs0=bcs_solid,
    bcs1=bcs_fluid,
    bcs_zero0=[],
    bcs_zero1=bcs_zero,
    dirichlet_interface=dirichlet_interface,
)

solver.snes.npc.setMonitor(my_monitor)


while t < t_end:
    Sys.Print(f"    t = {t}")
    inflow_expr.v = velocity(t)
    converged = False
    #PETSc.Options().setValue('npc_snes_lag_jacobian', -2)
    #solver.snes.setFromOptions()
    solver.solve()

    w0.assign(w)
    (v, u, p) = w.split(True)
    # save and plot
    u.rename("u", "u")
    v.rename("v", "v")
    p.rename("p", "p")

    xdmf_v.write(v, t)
    xdmf_u.write(u, t)
    xdmf_p.write(p, t)

    if u_init != None:
        xdmf_u_init.write(u_init, t)
    with df.HDF5File(comm, directory + "result.h5", 'a') as hdf_file:
        hdf_file.write(v, f"{n}/v")
        hdf_file.write(u, f"{n}/u")
        hdf_file.write(p, f"{n}/p")
        if u_init != None:
            hdf_file.write(u_init, f"{n}/u_init")
    if rank == 0:
        with open(directory + "checkpoint.txt", 'w') as ch_file:
            ch_file.write(f"{n} {t}")
    t += float(dt)
    n += 1
