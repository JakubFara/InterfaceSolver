from dolfin import (
    Mesh, FunctionSpace, Function, TestFunction, inner, grad, dx, cells,
    CompiledSubDomain, DirichletBC, Constant, XDMFFile, HDF5File, MeshFunction,
    dS, Expression
)
from mpi4py import MPI
from InterfaceSolver import NonlinearBrokenSolver


continuous = False  # run continuous or discontinuous problem
comm = MPI.COMM_WORLD
directory = 'mesh/'
name = 'broken_mesh'

# load the broken mesh
mesh = Mesh()
with HDF5File(mesh.mpi_comm(), f"{directory + name}.h5", "r") as hdf:
    hdf.read(mesh, "/mesh", False)
    dim = mesh.topology().dim()
    interface_entities = MeshFunction('bool', mesh, dim - 1, False)
    boundary_entities = MeshFunction('bool', mesh, dim - 2, False)
    hdf.read(interface_entities, "/interface")
    hdf.read(boundary_entities, "/boundary")

# we need to distinguish from which side discontinuity- we label cells under
# and behing the cranny
marker = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
for c in cells(mesh):
    if c.midpoint().y() > 0.5:
        marker[c] = 1

cell_val = 0  # bottom
top_sign = '-'
bottom_sign = '+'  # plus corresponds to the cell val

# define the space and functions
V = FunctionSpace(mesh, 'CG', 2)
u = Function(V)
v = TestFunction(V)

# boundary conditions
top = CompiledSubDomain("near(x[1], top) && on_boundary", top=1.0)
bottom = CompiledSubDomain("near(x[1], bottom) && on_boundary", bottom=0.0)
bcb = DirichletBC(V, Constant(0.0), bottom)
bct = DirichletBC(V, Constant((1.0)), top)
bcs = [bcb, bct]

# ufl form
a = inner(grad(u), grad(v))*dx
f = Expression('exp(-(x[0]-0.5)*(x[0]-0.5)-(x[1]-0.5)*(x[1]-0.5) )', degree=2)
lhs = -10*f*v*dx
a += lhs

# interface ufl form
n = Constant((0.0, 1.0))
Tn = inner(grad(u(top_sign)), n)

if not continuous:
    theta = 0.5
    a_interface = (
        - theta*inner(Tn, v(bottom_sign))*dS - 10*inner(f, v(bottom_sign))*dS
        + (1-theta)*inner((u(top_sign) - 1*u(bottom_sign)), v(bottom_sign))*dS
    )
    dirichlet_bcs = None
else:
    a_interface = (
        - inner(Tn, v(bottom_sign))*dS - 10*inner(f, v(bottom_sign))*dS
    )

    class Continuity():

        def __init__(self):
            pass

        def jacobian(self, coordinates, x1, x2):
            return [1, -1]

        def residual(self, coordinates, x1, x2):
            return x1[()] - x2[()]

    dirichlet_bcs = [
        ((), Continuity(), top_sign)
    ]

# create the solver
solver = NonlinearBrokenSolver(
    u, marker, interface_entities, boundary_entities, comm=None,
    interface_value=True, cell_val=cell_val, params=None, monitor=True
)

# solve
solver.solve(
    a, a_interface, bcs=bcs, bcs_zero=None, dirichlet_interface=dirichlet_bcs
)

# save the solution
u.rename('u', 'u')
directory = 'results/broken_laplace'
with XDMFFile(comm, f"{directory}/u.xdmf") as xdmf:
    xdmf.parameters["flush_output"] = True
    xdmf.parameters["functions_share_mesh"] = True
    xdmf.write(u)
