from dolfin import (
    Mesh, FiniteElement, VectorElement, MixedElement, MeshFunction, cells,
    FunctionSpace, TestFunction, Function, CompiledSubDomain, DirichletBC,
    Measure, Constant, Expression, inner, grad, split, exp, dS, XDMFFile
)
from mpi4py import MPI
from InterfaceSolver import NonlinearInterfaceSolver
from InterfaceSolver import interface


comm = MPI.COMM_WORLD
size = comm.Get_size()


def interface_func(x, y):
    return y-0.5


mesh = Mesh("mesh/mesh.xml")
marker = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
for c in cells(mesh):
    marker[c] = c.midpoint().y() > 0.5

cell_val = 0  # bottom
top_sign = '-'
bottom_sign = '+'  # minus corresponds to the cell val

# function spaces
E1 = FiniteElement("CG", mesh.ufl_cell(), 2)
E2 = FiniteElement("CG", mesh.ufl_cell(), 2)

V = FunctionSpace(mesh, MixedElement([E1, E2]))

v = TestFunction(V)
u = Function(V)
(v1, v2) = split(v)
(u1, u2) = split(u)

# boundary
top = CompiledSubDomain("near(x[1], top) && on_boundary", top=1.0)
bottom = CompiledSubDomain("near(x[1], bottom) && on_boundary", bottom=0.0)
middle = CompiledSubDomain("near(x[1], middle) ", middle=0.5)

bcb1 = DirichletBC(V.sub(0), Constant(-1.0), bottom)
bct1 = DirichletBC(V.sub(0), Constant((1.0)), top)
bcm1 = DirichletBC(V.sub(0), Constant((0.0)), middle)

bcb2 = DirichletBC(V.sub(1), Constant(-1.0), bottom)
bct2 = DirichletBC(V.sub(1), Constant((1.0)), top)
bcm2 = DirichletBC(V.sub(1), Constant((0.0)), middle)

# ufl form
interface = interface(mesh, interface_func, val=1)
dX = Measure("dx")(domain=mesh, subdomain_data=marker)

alpha1 = 1.0
n = Constant((0., 1.))
Tn1 = inner(grad(u1(top_sign)), n)

a1_interface = (
    inner((u1(top_sign) - 2*u2(bottom_sign)), v1(top_sign))*dS
    - inner(Tn1, v1(bottom_sign))*dS
)

alpha2 = 1.0
Tn2 = inner(grad(u2(top_sign)), n)

a2_interface = (
    # inner((u2(top_sign) - u1(bottom_sign)), v2(top_sign))*dS
    - inner(Tn2, v2(bottom_sign))*dS
)

a_interface = a1_interface + a2_interface

a1 = (
    inner(grad(v1), grad(u1))*dX(1)
    + inner(grad(v2), grad(u2))*dX(1)
    + exp(u1)*v1*dX(1)
)

a0 = (
    inner(grad(v1), grad(u1))*dX(0)
    + inner(grad(v2), grad(u2))*dX(0)
    + 2*exp(u1)*v1*dX(0)
)

f1 = Expression('exp(-(x[0]-0.5)*(x[0]-0.5)-(x[1]-0.5)*(x[1]-0.5) )', degree=2)
l1 = 10.0*f1*v1*dX(1) + 10.0*f1*v2*dX(1)
a1 += l1

f0 = Expression('exp(-(x[0]-0.5)*(x[0]-0.5)-(x[1]-0.5)*(x[1]-0.5) )', degree=2)
l0 = -10.0*f0*v1*dX(0) - 10.0*f0*v2*dX(0)
a0 += l0


class Discontinuity():

    def __init__(self):
        pass

    def jacobian(self, coordinates, x1, x2):
        # first cell_val ... here bottom
        # [u1_bottom, u2_bottom, u1_top, u2_top]
        return [1, 0, 0, -1]

    def residual(self, coordinates, x1, x2):
        # x1 - on area cell_val
        # x2 - on the remaining part
        u2_top = x2[(1, )]
        u1_bottom = x1[(0, )]
        return u1_bottom - u2_top


# we will wtire it as dirichlet to u1 on top
dirichlet_bcs = [
    ((1, ), Discontinuity(), top_sign)
]

# solve
Solver = NonlinearInterfaceSolver(u, marker, interface,
                                  interface_value=1, cell_val=cell_val)

# bcs_zero set 0 on rows in jacobian and replace them by
# 'interface dirichlet BCs'
Solver.solve(
    a0, a1, a_interface, bcs0=[bct1, bct2], bcs1=[bcb1, bcb2],
    bcs_zero1=[bcm1, bcm2], dirichlet_interface=dirichlet_bcs
)

# save and plot
u.rename('u', 'u')
directory = 'results/nonlinear_mixedspaces'
with XDMFFile(comm, f"{directory}/u.xdmf") as xdmf:
    xdmf.parameters["flush_output"] = True
    xdmf.parameters["functions_share_mesh"] = True
    xdmf.write(u)

# import matplotlib.pyplot as plt
# plot(u)
# plt.show()
