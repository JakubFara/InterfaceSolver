from mpi4py import MPI
from dolfin import parameters, UnitSquareMesh, cells, MeshFunction, info
from InterfaceSolver import make_discontinuous_mesh


comm = MPI.COMM_WORLD
size = comm.Get_size()

if size == 1:
    parameters["ghost_mode"] = "none"  
    mesh = UnitSquareMesh(20, 20,"crossed")
    directory = 'mesh/'
    name = 'mesh'
    cell_function_file = directory + 'cell_function.xml'
    mesh_path = directory + name + '.xml'
    marker = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    for c in cells(mesh):
        marker[c] = c.midpoint().y() < 0.5
    mesh = make_discontinuous_mesh(
        mesh, marker, 1, 0, save=True, directory=directory, name=name)
else:
    info('Run this file only in serial!!')