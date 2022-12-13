from dolfin import (
    Mesh, FunctionSpace, Function, TestFunction, inner, grad, dx, solve,
    CompiledSubDomain, DirichletBC, Constant, XDMFFile
)
from mpi4py import MPI


comm = MPI.COMM_WORLD
mesh = Mesh("mesh/broken_mesh.xml")
V = FunctionSpace(mesh, 'CG', 1)
v = Function(V)
u = TestFunction(V)

a = inner(grad(u), grad(v))*dx
top =  CompiledSubDomain("near(x[1], top) && on_boundary", top = 1.0)
bottom = CompiledSubDomain("near(x[1], bottom) && on_boundary", bottom = 0.0)
bcb = DirichletBC(V, Constant(0.0), bottom)
bct = DirichletBC(V, Constant((1.0)), top)
bcs = [bcb, bct]

solve(a==0, v, bcs = bcs)

# save and plot
v.rename('u','u')
directory = 'results/broken_laplace'
with XDMFFile(comm, f"{directory}/u.xdmf" ) as xdmf:
    xdmf.parameters["flush_output"] = True
    xdmf.parameters["functions_share_mesh"] = True
    xdmf.write(v)