import dolfin as df
from hdf_writer import H5FileWriter
from numpy import pi


mesh = df.Mesh()
comm = mesh.mpi_comm()
result_file = "result.h5"
result_file_v = "result_v"
result_file_u = "result_u"
result_file_u_init = "result_u_init"
# result_file_p = "results/tube3d/theta0.5/radius0.018/result.h5"
# with df.HDF5File(comm, folder + result_file, "r") as hdf_file:
#     hdf_file.read(mesh, "/mesh", False)
# theta = 0.5
theta = 1.0
maximal_radius = 0.016
mesh_level = 2


class Parameters:
    rho_s: float = 1.0e3
    nu_s: float = 0.49
    mu_s: float = 10000.0
    rho_f: float = 1.05e3
    mu_f: float = 3.8955e-3
    # mu_f: float = 1.0
    E: float = 2 * mu_s * (1 + nu_s)
    lam: float = E * nu_s / ((1 + nu_s) * (1 - 2 * nu_s))


parameters = Parameters()

folder = f"/usr/work/fara/aorta3d/theta{theta}/radius{maximal_radius}/g{parameters.mu_s}/ml{mesh_level}/"

labels = {
      "solid": 1,
      "fluid": 2,
      "inflow_s": 1,
      "inflow_f": 2,
      "outflow_s": 3,
      "outflow_f": 4,
      "out": 5,
      "interface": 6,
      "mid_f": 7,
      "mid_s": 8,
      "solid_sign": "+",
      "fluid_sign": "-",
}

comm = df.MPI.comm_world
with df.HDF5File(comm, f"data/aorta_refined_discontinuous_lev{mesh_level}_r{int(maximal_radius*1000)}.h5", "r") as h5_file:
# with df.HDF5File(comm, "data/aorta_discontinuous.h5", "r") as h5_file:
    # first we need to create an empty mesh
    mesh = df.Mesh(comm)
    # load the data stored in `/mesh` to the mesh
    h5_file.read(mesh, "/mesh", False)
    mesh2 = df.Mesh(mesh)
    # the dimension of the mesh
    # dim = mesh.geometry().dim()
    dim = 3
    # we need to create the empty meshfunctions at first
    cell_marker = df.MeshFunction('size_t', mesh, dim)
    bndry_marker = df.MeshFunction('size_t', mesh, dim - 1)
    interface_marker = df.MeshFunction('size_t', mesh, dim - 1)
    # we load the data to the subdomains markers
    h5_file.read(cell_marker, "/cell_marker")
    h5_file.read(bndry_marker, "/facet_marker")
    h5_file.read(interface_marker, "/interface_marker")


element_p = df.FiniteElement("CG", mesh.ufl_cell(), 1)
element_u = df.VectorElement("CG", mesh.ufl_cell(), 2)
element_v = df.VectorElement("CG", mesh.ufl_cell(), 2)

space_v = df.FunctionSpace(mesh, element_v)
space_u = df.FunctionSpace(mesh, element_u)
space_p = df.FunctionSpace(mesh, element_p)

df.info(f"{space_v.dim() + space_u.dim() + space_p.dim()}")

v = df.Function(space_v)
u = df.Function(space_u)
u_init = df.Function(space_u)
p = df.Function(space_p)

h5_file_v = H5FileWriter(result_file_v, comm, folder=folder)
h5_file_u = H5FileWriter(result_file_u, comm, folder=folder)
h5_file_u_init = H5FileWriter(result_file_u_init, comm, folder=folder)
# h5_file_p = H5FileWriter(result_file_read, comm)


def int_dx(integrand, dx, F):
    if F is not None:
        return df.assemble(integrand * df.det(F) * dx)
    else:
        return df.assemble(integrand * dx)


def int_ds(integrand, ds, F, n_ref, label=None):
    n_norm = df.sqrt(df.inner(df.inv(F).T * n_ref, df.inv(F).T * n_ref))
    # n = df.inv(F).T * n_ref / n_norm
    if label != None:
        return df.assemble(integrand * df.det(F) * n_norm * ds(label))
    else:
        return df.assemble(integrand * df.det(F) * n_norm * ds)


def extract_quantites(t, mesh, bndry_marks, cell_marks, labels, theta, v, u, p, n_ref,
                      mu, mu_s, gamma, filename='results/quantities.txt', r=0.02,
                      reset=False, u_init=None):

    identity = df.Identity(3)
    if u_init is not None:
        F = (
            (identity + df.grad(u) * df.inv(identity + df.grad(u_init)))
            * (identity + df.grad(u_init))
        )
    else:
        F = identity + df.grad(u)

    comm = df.MPI.comm_world
    n_norm = df.sqrt(df.inner(df.inv(F).T * n_ref, df.inv(F).T * n_ref))
    n = df.inv(F).T * n_ref / n_norm
    J = df.det(F)

    dx_f = df.Measure("dx", domain=mesh, subdomain_data=cell_marks, metadata={"quadrature_degree": 8})(labels["fluid"])
    dx_s = df.Measure("dx", domain=mesh, subdomain_data=cell_marks, metadata={"quadrature_degree": 8})(labels["solid"])

    file_xml = df.File("bndry_marker_temp.pvd")
    file_xml << bndry_marker

    ds = df.Measure("ds", subdomain_data=bndry_marks, domain=mesh, metadata={"quadrature_degree": 8})

    DG0 = df.FunctionSpace(mesh, df.FiniteElement("DG", mesh.ufl_cell(), 0))
    CG1 = df.FunctionSpace(mesh, df.FiniteElement("CG", mesh.ufl_cell(), 1))
    CG2 = df.FunctionSpace(mesh, df.FiniteElement("CG", mesh.ufl_cell(), 2))
    dx_v = df.grad(v) * df.inv(F)

    # curlv = df.Function(CG2)
    # curlv.assign(
    #     df.project(
    #         (
    #             dx_v[2, 1] - dx_v[1, 2]
    #             + dx_v[0, 2] - dx_v[2, 0]
    #             + dx_v[1, 0] - dx_v[0, 1]
    #         )
    #         , CG2, solver_type='mumps')
    # )
    curlv = (
        dx_v[2, 1] - dx_v[1, 2]
        + dx_v[0, 2] - dx_v[2, 0]
        + dx_v[1, 0] - dx_v[0, 1]
    )

    vt = v - df.inner(v, n) * n
    Dv = 0.5 * (df.grad(v) * df.inv(F) + df.inv(F).T * df.grad(v).T  )
    parameters_dict=dict()
    one = df.Constant(1.0)
    volume = int_dx(df.interpolate(one, DG0), dx_f, F)
    area_wall = int_ds(one, ds, F, n_ref, label=labels['mid_f'])
    area_out = int_ds(one, ds, F, n_ref, label=labels['inflow_f'])
    area_in = int_ds(one, ds, F, n_ref, label=labels['outflow_f'])
    df.info(f"area_in - {area_in} = area_in - {area_out}")

    # df.info(f"{area_wall} {area_out} {area_in}")
    parameters_dict['time'] = t
    parameters_dict['bulk_diss'] = (
        int_dx(2.0 * mu * df.inner(Dv, Dv), dx_f, F)
    )
    parameters_dict['solid_diss'] = (
        int_dx(2.0 * mu_s * df.inner(Dv, Dv), dx_s, F)
    )
    if theta != 1.0:
        parameters_dict['bndry_diss'] = (
            theta / (gamma * (1.0 - theta))
            * int_ds(df.inner(vt, vt), ds, F, n_ref, label=labels['mid_f'])
        )

        parameters_dict['WSScomp'] = (
            theta / (gamma * (1.0 - theta))
            * int_ds(df.sqrt(df.inner(vt, vt)), ds, F, n_ref,  label=labels['mid_f'])
            / area_wall
        )
    else:
        parameters_dict['bndry_diss'] = 0.0
    parameters_dict['total_diss'] = (
        parameters_dict['bulk_diss']
        + parameters_dict['bndry_diss']
    )

    parameters_dict['Pdrop'] =  (
        int_ds(p, ds, F, n_ref, label=labels['inflow_f']) / area_in
        - int_ds(p, ds, F, n_ref, label=labels['outflow_f']) / area_out
    )
    parameters_dict['Vort'] =  int_dx(abs(curlv), dx_f, F) / volume
    parameters_dict['normalvel'] = (
        int_ds(
            df.inner(v, n) * df.inner(v, n), ds, F, n_ref, label=labels['mid_f']
        )**(0.5)
    )
    # u_x, u_y = u.split(True)
    # v_x, v_y = v.split(True)
    # u_norm = df.Function(u_x.function_space())
    # v_norm = df.Function(v_x.function_space())
    # u_norm.vector()[:] = np.sqrt(u_x.vector() * u_x.vector() + u_y.vector() * u_y.vector()+ u_z.vector() * u_z.vector())
    # v_norm.vector()[:] = np.sqrt(v_x.vector() * v_x.vector() + v_y.vector() * v_y.vector()+ v_z.vector() * v_z.vector())

    # vec = df.as_backend_type(v.vector()).vec()
    # # parameters_dict['v_max'] = vec.max()[1]
    # parameters_dict['v_max_s'] = maximum(v_norm, cell_marks, 2, comm)
    # parameters_dict['u_max_x'] =  maximum(u_x, cell_marks, 2, comm)  # df.as_backend_type(u_x.vector()).vec().max()[1]
    # parameters_dict['u_max_s'] = maximum(u_norm, cell_marks, 2, comm)  # df.as_backend_type(u.vector()).vec().max()[1]
    # # df.info(f"{parameters_dict['v_max_s']}")
    # parameters_dict['u_mid'] = peval(u, (r + 0.001, 0.0))[0]
    if mesh.mpi_comm().rank == 0:
        result_string = ''
        writing = 'a'
        if reset:
            for name in parameters_dict.keys():
                result_string += f'{name} '
            result_string += '\n'
            writing = 'w'
        for value in parameters_dict.values():
            result_string += f'{value:.8f} '
        with open(filename, writing) as result_file:
            result_file.write(result_string + '\n')
    return parameters_dict


with df.XDMFFile("p.xdmf") as xdmf_file_p:
    pass

dt = 0.005
t = 0
gamma=3.08
reset = True
for n in range(10000):
    df.info(f"{n}")
    t += dt
    with df.HDF5File(comm, folder + result_file, "r") as hdf_file:
        hdf_file.read(v, f"{n}/v")
        hdf_file.read(u, f"{n}/u")
        hdf_file.read(u_init, f"{n}/u_init")
        hdf_file.read(p, f"{n}/p")
    # u_init = None

    n_ref = df.FacetNormal(mesh)
    extract_quantites(
        t, mesh, bndry_marker, cell_marker, labels, theta, v, u, p, n_ref,
        parameters.mu_f, parameters.mu_s, gamma,
        filename=f'results/quantities_theta{theta}_r{maximal_radius}_lev{mesh_level}.txt', r=0.02,
        reset=reset, u_init=u_init
    )
    reset = False

    # h5_file_v.write(v, mesh, time=t)
    # h5_file_u.write(u, mesh, time=t)
    # h5_file_u_init.write(u_init, mesh, time=t)
    p.rename("p", "p")
    df.info(f"t = {t}")
    xdmf_file_p.write(p, t)

    # h5_file_p.write(p, mesh, time=t)
