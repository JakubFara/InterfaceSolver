from mpi4py import MPI
from dolfin import (
    parameters, UnitSquareMesh, cells, MeshFunction, info, UnitCubeMesh
)
from InterfaceSolver import make_discontinuous_mesh


comm = MPI.COMM_WORLD
size = comm.Get_size()
dim = 3
resolution = 20

if size == 1:
    parameters["ghost_mode"] = "none"
    if dim == 2:
        mesh = UnitSquareMesh(20, 20, "crossed")
    elif dim == 3:
        mesh = UnitCubeMesh(20, 20, 20)
    directory = 'mesh/'
    if dim == 2:
        name = 'mesh'
    else:
        name = f'mesh3d'
    cell_function_file = directory + 'cell_function.xml'
    mesh_path = directory + name + '.xml'
    marker = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    for c in cells(mesh):
        marker[c] = c.midpoint().y() < 0.5
    mesh = make_discontinuous_mesh(
        mesh, marker, 1, 0, save=True, directory=directory, name=name)
else:
    info('Run this file only in serial!!')
