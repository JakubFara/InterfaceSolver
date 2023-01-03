from dolfin import (
    parameters, Mesh, MeshFunction, cells, FunctionSpace, TestFunction,
    Function, CompiledSubDomain, DirichletBC, Measure, Constant, Expression,
    inner, grad, dS, XDMFFile, FiniteElement, MixedElement, VectorElement,
    split, div, sym, Identity
)
from mpi4py import MPI
from InterfaceSolver import NonlinearInterfaceSolver
from InterfaceSolver import interface


comm = MPI.COMM_WORLD
size = comm.Get_size()

parameters["ghost_mode"] = "none"
# load the discontinuous mesh. Make sure you build that -> python make_mesh.py
mesh = Mesh("mesh/tube3d.xml")

# label the top and the bottom subdomains
marker = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
radius = 0.2
radius2 = 0.3
for c in cells(mesh):
    r = c.midpoint().x()**2 + c.midpoint().y()**2
    if r > radius**2:
        marker[c] = 1

cell_val = 0  # bottom
inner_sign = '+'  # plus corresponds to the cell val
outer_sign = '-'

# function spaces
Ep = FiniteElement("CG", mesh.ufl_cell(), 1)
Ev = VectorElement("CG", mesh.ufl_cell(), 2)

V = FunctionSpace(mesh, MixedElement([Ev, Ep]))
w = Function(V)
w_ = TestFunction(V)
(v, p) = split(w)
(v_, p_) = split(w_)

# Boundary
inflow = CompiledSubDomain(
    "near(x[2], 0.0) && on_boundary"
)
inflow_inner = CompiledSubDomain(
    (
        "near(x[2], 0.0) && pow(x[0], 2) + pow(x[1], 2) <= pow(r, 2) + eps"
        "&& on_boundary"
    ),
    r=radius, eps=0.00001
)
inflow_outer = CompiledSubDomain(
    (
        "near(x[2], 0.0) && pow(x[0], 2) + pow(x[1], 2) > pow(r, 2) - eps "
        "&& on_boundary"
    ),
    r=radius, eps=0.00001
)
outflow_inner = CompiledSubDomain(
    (
        "near(x[2], 2.0) && pow(x[0], 2) + pow(x[1], 2) <= pow(r, 2) + eps "
        "&& on_boundary"
    ),
    r=radius, eps=0.00001
)
outflow_outer = CompiledSubDomain(
    (
        "near(x[2], 2.0) && pow(x[0], 2) + pow(x[1], 2) > pow(r, 2) "
        "&& on_boundary"
    ),
    r=radius
)
cylinder = CompiledSubDomain(
    "pow(x[0], 2) + pow(x[1], 2) >= pow(r, 2) - eps && on_boundary",
    r=radius2, eps=0.01
)

middle = CompiledSubDomain((
    "pow(x[0], 2) + pow(x[1], 2) > pow(r, 2) - eps && "
    "pow(x[0], 2) + pow(x[1], 2) < pow(r, 2) + eps "),
    eps=0.003, r=radius)

inflow_expr = Expression(
    ("0.0", "0.0", "-pow(x[0], 2) - pow(x[1], 2) + pow(r, 2)"),
    r=radius, degree=2)

zero_vec = Constant((0.0, 0.0, 0.0))
bc_in0 = DirichletBC(V.sub(0), inflow_expr, inflow_inner)
bc_in1 = DirichletBC(V.sub(0), zero_vec, inflow_outer)

bc_out = DirichletBC(V.sub(0), zero_vec, outflow_inner)
bc_cylinder = DirichletBC(V.sub(0), zero_vec, cylinder)
bcm = DirichletBC(V.sub(0), zero_vec, middle)


def interface_func(x, y, z):
    return x**2 + y**2 - radius**2


interface = interface(mesh, interface_func, val=1, eps=0.003)
dX = Measure("dx")(domain=mesh, subdomain_data=marker)

# ufl forms
# normal vector
n = Expression(
    (
        'x[0]/pow(pow(x[0], 2) + pow(x[1], 2), 0.5)',
        'x[1]/pow(pow(x[0], 2) + pow(x[1], 2), 0.5)',
        '0.0'
    ),
    degree=4
)
mu1 = 1.0
mu0 = 2.0
T = mu0*sym(grad(v(outer_sign))) - p(outer_sign)*Identity(3)
Tn = T*n

a_interface = (
    inner((v(inner_sign) - 2*v(outer_sign)), v_(outer_sign))*dS
    - 1.0*inner(Tn, v_(inner_sign))*dS
)

a1 = mu1*inner(grad(v), grad(v_))*dX(1) - div(v_)*p*dX(1) + div(v)*p_*dX(1)
a0 = mu0*inner(grad(v), grad(v_))*dX(0) - p*div(v_)*dX(0) + div(v)*p_*dX(0)

# right-hand side
# f1 = Expression(
#     'exp(-(x[0]-0.5)*(x[0]-0.5)-(x[1]-0.5)*(x[1]-0.5))',
#     degree=2
# )
# l1 = 10.0*f1*v*dX(1)
# a1 += l1

# f0 = Expression(
#     'exp(-(x[0]-0.5)*(x[0]-0.5)-(x[1]-0.5)*(x[1]-0.5))', degree=2
# )
# l0 = -10.0*f0*v*dX(0)
# a0 += l0

# solve
solver = NonlinearInterfaceSolver(
    w, marker, interface, interface_value=1, cell_val=cell_val, params=None)

solver.solve(
    a0, a1, a_interface, bcs0=[bc_in0], bcs1=[bc_in1, bc_cylinder],
    bcs_zero0=[], bcs_zero1=[bcm]
)

(v, p) = w.split(True)
# save and plot
directory = 'results/tube3d'
with XDMFFile(comm, f"{directory}/v.xdmf") as xdmf:
    xdmf.parameters["flush_output"] = True
    xdmf.parameters["functions_share_mesh"] = True
    xdmf.write(v)
