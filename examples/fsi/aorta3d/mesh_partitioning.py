import dolfin as df
import numpy as np
import matplotlib.pyplot as plt


self_comm = df.MPI.comm_self
comm = df.MPI.comm_world

code = """
#include <dolfin/mesh/MeshPartitioning.h>
#include <dolfin/mesh/LocalMeshData.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/graph/SCOTCH.h>

#include <dolfin/mesh/CellType.h>
#include <dolfin/mesh/DistributedMeshTools.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/LocalMeshData.h>


#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
namespace dolfin {
    void build_distributed_mesh(std::shared_ptr<Mesh> mesh, std::vector<int> cell_part)
    {
        // Create and distribute local mesh data
        LocalMeshData local_mesh_data(*mesh);
        // MeshPartitioning::build_distributed_mesh(*mesh, local_mesh_data, parameters["ghost_mode"]);
        // Attach cell destinations
        // local_mesh_data.topology.cell_partition = cell_destinations;
        // std::vector<int> cell_part = local_mesh_data.topology.cell_partition;
        local_mesh_data.topology.cell_partition = cell_part;
        // std::vector<int>::iterator ptr;
        // for (ptr = cell_part.begin(); ptr < cell_part.end(); ptr++)
        //    cout << *ptr << " ";
        // MeshPartitioning::build_distributed_mesh(*mesh, local_mesh_data, parameters["ghost_mode"]);
        // return cell_part;
        // MeshPartitioning::build_distributed_mesh(*mesh);
    }


void build_distributed_mesh2(std::shared_ptr<Mesh> mesh,std::shared_ptr<Mesh> mesh2,
                                              const std::string ghost_mode)
{
  MPI_Comm comm = (*mesh).mpi_comm();
  if (MPI::size((*mesh).mpi_comm()) > 1){
    LocalMeshData mesh_data2(*mesh2);
    LocalMeshData mesh_data(*mesh);
    // Get mesh partitioner
    const std::string partitioner = parameters["mesh_partitioner"];

    // MPI communicator

    // Compute cell partitioning or use partitioning provided in local_data
    std::vector<int> cell_partition;
    std::map<std::int64_t, std::vector<int>> ghost_procs;
    // MeshPartitioning::partition_cells(comm, mesh_data2, partitioner, cell_partition, ghost_procs);
    std::unique_ptr<CellType> cell_type(CellType::create(mesh_data2.topology.cell_type));
    SCOTCH::compute_partition(comm, cell_partition, ghost_procs,
                              mesh_data2.topology.cell_vertices,
                              mesh_data2.topology.cell_weight,
                              mesh_data2.geometry.num_global_vertices,
                              mesh_data2.topology.num_global_cells,
                              *cell_type);
    MeshPartitioning::build_distributed_mesh(*mesh,
                            cell_partition,
                            ghost_mode);
  }
}



};
PYBIND11_MODULE(SIGNATURE, m)
{
    m.def("build_distributed_mesh", dolfin::build_distributed_mesh),
    m.def("build_distributed_mesh2", dolfin::build_distributed_mesh2);
}
"""


def get_mesh_on_root(comm, meshfile):
    #m = Mesh()
    mesh_global = df.Mesh(comm)
    # dim = mesh_global.dim()
    dim = 3

    self_comm = df.MPI.comm_self
    mesh = df.Mesh(self_comm)
    suffix = meshfile.split('.')[-1]
    if suffix == 'xdmf':
        with df.XDMFFile(self_comm, meshfile) as xdmf_file:
            xdmf_file.read(mesh)
    elif suffix == 'h5':
        with df.HDF5File(self_comm, meshfile, 'r') as xdmf_file:
            xdmf_file.read(mesh, '/mesh', False)
    coor = mesh.coordinates()
    cells = np.array([c.entities(0) for c in df.cells(mesh)])
    if comm.Get_rank() == 0:
        editor = df.MeshEditor()
        n_coords = coor.shape[0]
        n_cls = cells.shape[0]

        editor.open(mesh_global, 'tetrahedron', 3, 3)  # top. and geom. dimension are both 2
        editor.init_vertices(n_coords)  # number of vertices
        editor.init_cells(n_cls)     # number of cells
        for i in range(n_coords):
            editor.add_vertex(i, coor[i])
        for i in range(n_cls):
            if cells[i, 0] >= 0 and cells[i, 1] >=0 and cells[i, 2] >=0:
                editor.add_cell(i, cells[i])
            else:
                print("wrong cell", cells[i])
        editor.close()

    else:
        editor = df.MeshEditor()
        editor.open(mesh_global,'tetrahedron', 3, 3)  # top. and geom. dimension are both 2
        editor.init_vertices(4)  # number of vertices
        editor.init_cells(1)     # number of cells
        editor.add_vertex(0, [0, 0, 0])
        editor.add_vertex(1, [1, 0, 0])
        editor.add_vertex(2, [0, 1, 0])
        editor.add_vertex(3, [0, 0, 1])

        editor.add_cell(0, [0, 1, 2, 3])
        editor.close()

    return mesh_global

def get_partitioned_mesh(disc_mesh_file, cont_mesh_file):
    mesh_on_root = get_mesh_on_root(comm, cont_mesh_file)
    disc_mesh_on_root = get_mesh_on_root(comm, disc_mesh_file)
    mod = df.compile_cpp_code(code)
    mod.build_distributed_mesh2(disc_mesh_on_root, mesh_on_root, "none")
    return disc_mesh_on_root

# mesh = get_partitioned_mesh("aorta_refined_discontinuous_lev1_r16.h5", "aorta_lev1_r16.h5")

# df.plot(mesh)
# plt.show()

# V = df.FunctionSpace(mesh, 'CG', 1)

# # Define boundary condition
# # u_D = df.Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
# marker = df.MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
# for c in df.cells(mesh):
#     marker[c] = c.midpoint().y() < 0.5
# dX = df.Measure("dx")(domain=mesh, subdomain_data=marker)

# def boundary(x, on_boundary):
#     return on_boundary

# top =  df.CompiledSubDomain("near(x[1], top) && on_boundary", top = 1.0)
# bottom = df.CompiledSubDomain("near(x[1], bottom) && on_boundary", bottom = 0.0)


# bc = [
#     df.DirichletBC(V, "0.0", top),
#     df.DirichletBC(V, "1.0", bottom),
# ]

# # Define variational problem
# u = df.TrialFunction(V)
# v = df.TestFunction(V)
# f = df.Constant(-6.0)
# a = df.dot(df.grad(u), df.grad(v)) * dX(0)
# a += 2 * df.dot(df.grad(u), df.grad(v)) * dX(1)
# L = f * v * dX(0)
# L += 20 * f * v * dX(1)
# L2 = df.dot(df.grad(u("-")), df.grad(v("+"))) * df.dS

# # Compute solution
# u = df.Function(V)
# df.solve(a == L, u, bc)

# with df.XDMFFile("solution.xdmf") as xdmf_file:
#     xdmf_file.write(u)

# df.File("mesh_partitioning.pvd") << mesh
