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


def interface_func(x, y):
    return y-0.5


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

# ufl forms
interface = interface(mesh, interface_func, val=1)
dX = Measure("dx")(domain=mesh, subdomain_data=marker)

n = Constant((0., 1.))
# n = FacetNormal(mesh_interface)
Tn = inner(grad(u(bottom_sign)), n)
a_interface = (
    inner(Tn, v(top_sign))*dS
)

a1 = inner(grad(v), grad(u))*dX(1)
a0 = inner(grad(v), grad(u))*dX(0)

f1 = Expression('-60*pow(x[0]-0.5, 2)-pow(x[1]-0.5, 2)', degree=2)
l1 = f1*v*dX(1)
a1 += l1

f0 = Expression('-pow(x[0]-0.5, 2)-pow(x[1]-0.5, 2)', degree=2)
l0 = f0*v*dX(0)
a0 += l0


class Discontinuity():

    def __init__(self):
        pass

    def jacobian(self, coordinates, x1, x2):
        return [4*x1[()], -1]

    def residual(self, coordinates, x1, x2):
        return 2*x1[()]*x1[()] - x2[()]


dirichlet_bcs = [
    ((), Discontinuity(), bottom_sign)
]

params = {
    'snes_': {
        'rtol': 1.e-10,
        'atol': 1.e-10,
        'stol': 1.e-10,
        'max_it': 40
    }
}
# solve
Solver = NonlinearInterfaceSolver(
    u, marker, interface, interface_value=1, cell_val=cell_val, params=None)

Solver.solve(
    a0, a1, a_interface, bcs0=[bcb, bct], bcs1=[bct, bcb],
    bcs_zero0=[bcm], bcs_zero1=[], dirichlet_interface=dirichlet_bcs
)

# save and plot
directory = 'results/pointwise_dirichlet'
with XDMFFile(comm, f"{directory}/u.xdmf") as xdmf:
    xdmf.parameters["flush_output"] = True
    xdmf.parameters["functions_share_mesh"] = True
    f = xdmf

u.rename('u', 'u')
f.write(u)

# import matplotlib.pyplot as plt
# plot(u)
# plt.show()
