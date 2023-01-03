from dolfin import (
    Mesh, MeshFunction, cells, FunctionSpace, TestFunction, TrialFunction,
    Function, CompiledSubDomain, DirichletBC, Measure, Constant, Expression,
    inner, grad, dS, XDMFFile
)
from mpi4py import MPI
from InterfaceSolver import LinearInterfaceSolver
from InterfaceSolver import interface


comm = MPI.COMM_WORLD
size = comm.Get_size()

mesh = Mesh("mesh/mesh.xml")
marker = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
# mark cells ... 1 bottom ; 0 top
for c in cells(mesh):
    marker[c] = c.midpoint().y() > 0.5

cell_val = 0  # bottom
top_sign = '-'
bottom_sign = '+'  # minus corresponds to the cell val

# function space
V = FunctionSpace(mesh, 'CG', 1)
v = TestFunction(V)
u = TrialFunction(V)
sol = Function(V)

# boundary
top = CompiledSubDomain("near(x[1], top) && on_boundary", top=1.0)
bottom = CompiledSubDomain("near(x[1], bottom) && on_boundary", bottom=0.0)
middle = CompiledSubDomain("near(x[1], middle) ", middle=0.5)

bcb = DirichletBC(V, Constant(0.0), bottom)
bct = DirichletBC(V, Constant((1.0)), top)
bcm = DirichletBC(V, Constant((0.0)), middle)


def interface_func(x, y):
    return y-0.5


interface_marker = interface(mesh, interface_func, val=1)
dX = Measure("dx")(domain=mesh, subdomain_data=marker)

# ulf form
n = Constant((0., 1.))
# n = FacetNormal(mesh_interface)
Tn = inner(grad(u(top_sign)), n)
a_interface = (
    inner((u(top_sign) - 2*(u(bottom_sign))), v(top_sign))*dS
    - inner(Tn, v(bottom_sign))*dS
)

a1 = inner(grad(u), grad(v))*dX(1)  # bottom
a0 = inner(grad(u), grad(v))*dX(0)  # top

f1 = Expression('exp(-(x[0]-0.5)*(x[0]-0.5)-(x[1]-0.5)*(x[1]-0.5) )', degree=2)
l1 = 10.0*f1*v*dX(1)

f0 = Expression('exp(-(x[0]-0.5)*(x[0]-0.5)-(x[1]-0.5)*(x[1]-0.5) )', degree=2)
l0 = -10.0*f0*v*dX(0)

# solve
solver = LinearInterfaceSolver(
    sol, marker, interface_marker, interface_value=1, cell_val=cell_val)

solver.solve(a0, a1, a_interface, l0=l0, l1=l1, l_interface=None,
             bcs0=[bct], bcs1=[bcb], bcs_zero1=[bcm])

# save and plot
sol.rename('u', 'u')
directory = 'results/linear_parabolic'
with XDMFFile(comm, f"{directory}/u.xdmf") as xdmf:
    xdmf.parameters["flush_output"] = True
    xdmf.parameters["functions_share_mesh"] = True
    xdmf.write(sol)

# import matplotlib.pyplot as plt
# plot(sol)
# plt.show()
