from mpi4py import MPI
from dolfin import (
    parameters, UnitSquareMesh, cells, MeshFunction, entities, UnitCubeMesh,
    Mesh, XDMFFile
)
from InterfaceSolver import make_broken_mesh


comm = MPI.COMM_WORLD
size = comm.Get_size()

val = 1
if size == 1:
    parameters["ghost_mode"] = "none"
    mesh = UnitSquareMesh(10, 10, "crossed")
    dim = mesh.topology().dim()
    interface = MeshFunction('size_t', mesh, dim - 1, 0)
    # label the interface edges
    for edge in entities(mesh, dim - 1):
        x = edge.midpoint().x()
        y = edge.midpoint().y()
        if y == 0.5 and 0.2 < x < 0.6:
            interface[edge] = val

    make_broken_mesh(
        mesh, interface, val, directory='./mesh/', name='broken_mesh'
    )
