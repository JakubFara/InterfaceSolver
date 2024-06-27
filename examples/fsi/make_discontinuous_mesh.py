from InterfaceSolver import make_discontinuous_mesh
import dolfin as df
import json


# define mpi communicator
comm = df.MPI.comm_world
# run this script in serial
mesh_file_continuous = "data/tube3d.h5"
mesh_file_discontinuous = "data/tube3d_discontinuous.h5"


with df.HDF5File(comm, mesh_file_continuous, "a") as h5_file:
    # first we need to create an empty mesh
    continuous_mesh = df.Mesh()
    # load the data stored in `/mesh` to the mesh
    h5_file.read(continuous_mesh, "/mesh", False)
    # the dimension of the mesh
    dim = continuous_mesh.geometry().dim()
    # we need to create the empty meshfunctions at first
    cell_marker = df.MeshFunction('size_t', continuous_mesh, dim)
    facet_marker = df.MeshFunction('size_t', continuous_mesh, dim - 1)
    # we load the data to the subdomains markers
    h5_file.read(cell_marker, "/subdomains")
    # h5_file.read(facet_marker, "/facet_marker")


def hash_point_to_string(point: df.Point, dim, num_dig=5):
    mid = [point.x(), point.y()]
    mid_string = f"{mid[0]:.5f},{mid[1]:.5f}"
    if dim == 3:
        mid_string += f",{point.z():.5f}"
    return mid_string


def make_discontinuous_marker(mesh, discontinuous_mesh, marker, cell_type):
    dim = mesh.topology().dim()
    if cell_type == "cell":
        entities = df.cells(mesh)
        discontinuous_entities = df.cells(discontinuous_mesh)
        entity_dim = dim
    elif cell_type == "facet":
        entities = df.facets(mesh)
        discontinuous_entities = df.facets(discontinuous_mesh)
        entity_dim = dim - 1
    print(f"dim = {dim} entity dim = {entity_dim}")
    entities_dict = {}
    for entity in entities:
        mark = marker[entity]
        entities_dict[hash_point_to_string(entity.midpoint(), dim)] = mark
    # print(entities_dict)
    discontinuous_marker = df.MeshFunction('size_t', discontinuous_mesh, entity_dim)
    for entity in discontinuous_entities:
        mark = entities_dict[hash_point_to_string(entity.midpoint(), dim)]
        discontinuous_marker[entity] = mark
    return discontinuous_marker

discontinuous_mesh = make_discontinuous_mesh(
    continuous_mesh, cell_marker, 2, 1, save=False
)

discontinuous_cell_marker = make_discontinuous_marker(
    continuous_mesh, discontinuous_mesh, cell_marker, "cell"
)

with df.HDF5File(comm, mesh_file_discontinuous, "w") as h5_file:
    h5_file.write(discontinuous_mesh, "/mesh")
    h5_file.write(discontinuous_cell_marker, "/cell_marker")
    # h5_file.write(discontinuous_facet_marker, "/facet_marker")


comm = df.MPI.comm_world
# we load the mesh and the subdomains
with df.HDF5File(comm, mesh_file_discontinuous, "a") as h5_file:
    # first we need to create an empty mesh
    mesh = df.Mesh(comm)
    # load the data stored in `/mesh` to the mesh
    h5_file.read(mesh, "/mesh", False)
    # the dimension of the mesh
    dim = 3
    # we need to create the empty meshfunctions at first
    cell_marker = df.MeshFunction('size_t', mesh, dim)
    facet_marker = df.MeshFunction('size_t', mesh, dim - 1)
    # we load the data to the subdomains markers
    h5_file.read(cell_marker, "/cell_marker")
