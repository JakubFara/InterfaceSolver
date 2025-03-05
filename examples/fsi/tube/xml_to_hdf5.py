import dolfin as df
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--mesh_level" , "-ml",
    help=(
        "navire slip parameter",
    ),
    default=1,
    type=int,
)

args = vars(parser.parse_args())
mesh_level = args["mesh_level"]

mesh_lev = 1
mesh_name = f"tube3d_lev{mesh_level}"
mesh = df.Mesh(f"data/{mesh_name}.xml")
domains=mesh.domains()

if os.path.isfile(f"data/{mesh_name}_physical_region.xml"):
    domains.init(2)
    subdomains = df.MeshFunction("size_t", mesh, f"data/{mesh_name}_physical_region.xml")
boundaries = None
if os.path.isfile(f"data/{mesh_name}_facet_region.xml"):
    boundaries = df.MeshFunction("size_t", mesh, f"data/{mesh_name}_facet_region.xml")


with df.HDF5File(mesh.mpi_comm(), f"data/{mesh_name}.h5", "w") as hdf5_file:
    hdf5_file.write(mesh, "/mesh")
    hdf5_file.write(subdomains, "/subdomains")
    if boundaries is not None:
        hdf5_file.write(boundaries, "/boundaries")
    df.File("data/subdomains.pvd") << subdomains
    df.File("data/boundaries.pvd") << boundaries
