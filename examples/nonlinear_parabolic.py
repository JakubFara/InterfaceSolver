from dolfin import (
    parameters, Mesh, MeshFunction, cells, FunctionSpace, TestFunction,
    Function, CompiledSubDomain, DirichletBC, Measure, Constant, Expression,
    inner, grad, dS, XDMFFile
)
from mpi4py import MPI
from InterfaceSolver import NonlinearInterfaceSolver
from InterfaceSolver import interface


comm = MPI.COMM_WORLD
size = comm.Get_size()

parameters["ghost_mode"] = "none"
# load the discontinuous mesh. Make sure you build that -> python make_mesh.py
mesh = Mesh("mesh/mesh.xml")

# label the top and the bottom subdomains
marker = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
for c in cells(mesh):
    if c.midpoint().y() > 0.5:
        marker[c] = 1

cell_val = 0  # bottom
top_sign = '-'
bottom_sign = '+'  # plus corresponds to the cell val

# function spaces
V = FunctionSpace(mesh, 'CG', 2)

v = TestFunction(V)
u = Function(V)

# Boundary
top = CompiledSubDomain("near(x[1], top) && on_boundary", top=1.0)
bottom = CompiledSubDomain("near(x[1], bottom) && on_boundary", bottom=0.0)
middle = CompiledSubDomain("near(x[1], middle) ", middle=0.5)

bcb = DirichletBC(V, Constant(0.0), bottom)
bct = DirichletBC(V, Constant((1.0)), top)
bcm = DirichletBC(V, Constant((0.0)), middle)


def interface_func(x, y):
    return y-0.5


interface = interface(mesh, interface_func, val=1)
dX = Measure("dx")(domain=mesh, subdomain_data=marker)


# ufl forms
def gamma(u, p, epsilon=1.0e-7):
    value = (epsilon**2 + inner(grad(u), grad(u)))**((p - 2) / 2)
    return value


p0 = 1.8
p1 = 2.1
n = Constant((0., 1.))  # normal vector

Tn = gamma(u(top_sign), p1)*inner(grad(u(top_sign)), n)
a_interface = (
    inner((u(bottom_sign) - 1*u(top_sign)), v(top_sign))*dS
    - 1.0*inner(Tn, v(bottom_sign))*dS
)

a1 = inner(grad(v), gamma(u, p1)*grad(u))*dX(1)
a0 = inner(grad(v), gamma(u, p0)*grad(u))*dX(0)

# right-hand side
f1 = Expression('exp(-(x[0]-0.5)*(x[0]-0.5)-(x[1]-0.5)*(x[1]-0.5) )', degree=2)
l1 = 10.0*f1*v*dX(1)
a1 += l1

f0 = Expression('exp(-(x[0]-0.5)*(x[0]-0.5)-(x[1]-0.5)*(x[1]-0.5) )', degree=2)
l0 = -10.0*f0*v*dX(0)
a0 += l0

# solve
solver = NonlinearInterfaceSolver(
    u, marker, interface, interface_value=1, cell_val=cell_val, params=None)

solver.solve(a0, a1, a_interface,
             bcs0=[bct], bcs1=[bcb], bcs_zero0=[], bcs_zero1=[bcm])

# save and plot
directory = 'results/nonlinear_parabolic'
with XDMFFile(comm, f"{directory}/u.xdmf") as xdmf:
    xdmf.parameters["flush_output"] = True
    xdmf.parameters["functions_share_mesh"] = True
    f = xdmf

u.rename('u', 'u')
f.write(u)

# import matplotlib.pyplot as plt
# plot(u)
# plt.show()
