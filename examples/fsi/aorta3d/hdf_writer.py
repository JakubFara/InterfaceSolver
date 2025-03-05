import dolfin as df
import h5py


class H5FileWriter():
    def __init__(self, filename, comm, folder=None):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.filename = filename
        if folder is None:
            self.folder = ""
        else:
            self.folder = folder
            if self.folder[-1] != "/":
                self.folder += "/"
        self.step = 0
        self._initialize_file()

    def _initialize_file(self):
        if self.rank == 0:
            xdmf = (
                f'<?xml version="1.0" ?>\n'
                f'<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n'
                f'<Xdmf Version="2.0" xmlns:xi="[http://www.w3.org/2001/XInclude]">\n'
                f'   <Domain>\n'
                f'      <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">\n'
                f'      </Grid>\n'
                f'    </Domain>\n'
                f'</Xdmf>'
            )
            with open(f"{self.folder}{self.filename}.xdmf", "w") as xdmf_file:
                xdmf_file.write(xdmf)

    def write(self, func: df.Function, mesh: df.Mesh, time=0.0):
        space = func.function_space()
        dim = mesh.topology().dim()
        if dim == 2:
            topo_type = "Triangle"
            values_per_cell = 6
        elif dim == 3:
            topo_type = "Tetrahedron"
            values_per_cell = 10
        dofmap = space.dofmap()
        sub_functions = func.split(True)
        num_values = func.vector()[:].size
        if sub_functions == ():
            # num_values = func.vector()[:].size
            is_vec = False
        else:
            # num_values = sub_functions[0].vector()[:].size
            is_vec = True
        num_vertices = len([v for v in df.vertices(mesh)])
        cell_dict = {}

        for cell in df.cells(mesh):
            cell_dict[cell.index()] = dofmap.cell_dofs(cell.index())

        num_cells = len(cell_dict)

        for cell in df.cells(mesh):
            values_per_cell = len(dofmap.cell_dofs(cell.index()))
            break

        if self.step == 0:
            with df.HDF5File(self.comm, f"{self.folder}{self.filename}.h5", "w") as hdf5_file:
                hdf5_file.write(mesh, f"{self.step}/mesh")
        else:
            with df.HDF5File(self.comm, f"{self.folder}{self.filename}.h5", "a") as hdf5_file:
                hdf5_file.write(mesh, f"{self.step}/mesh")

        file_h5 = h5py.File(f"{self.folder}{self.filename}.h5", 'a', driver='mpio', comm=self.comm)
        indices_group = file_h5.create_group(f'{self.step}/indices')
        values_group = file_h5.create_group(f'{self.step}/values')

        dofs_in_cells = indices_group.create_dataset(
            'values_indices', (num_cells, values_per_cell), dtype='int'
        )

        # if func.split(True) == ():
        values = values_group.create_dataset(
            'vector', (num_values), dtype='float'
        )
        values[[i for i in range(num_values)]] = list(func.vector()[:])
        # else:
        #     values = values_group.create_dataset(
        #         'vector', (num_values, dim), dtype='float'
        #     )
        #     for j, sub_functions in enumerate(func.split(True)):
        #         values[[i for i in range(num_values)], j] = list(sub_functions.vector()[:])


        dofs_in_cells[list(cell_dict.keys())] = list(cell_dict.values())
        file_h5.close()
        self.create_xdmf(
            num_cells, num_vertices, num_values, values_per_cell,
            dim, is_vec, self.step, time=time, grid_name="grid", topo_type=topo_type
        )
        self.step += 1

    def create_xdmf(self, num_cells, num_vertices, num_values, values_per_cell, dim, is_vec, step, time=0.0, grid_name="grid", topo_type="Triangle"):
        format_type="HDF"
        element="CG"
        if dim == 2:
            geom_type = "XY"
            element_cell = "triangle"
        elif dim == 3:
            geom_type = "XYZ"
            element_cell = "tetrahedron"

        func_shape = f"{num_values}"
        if is_vec is True:
            # func_shape = f"{num_values}, {dim}"
            func_type = "Vector"
        else:
            func_type = "Scalar"


        xdmf = (
            f'        <Grid Name="{grid_name}">\n'
            f'           <Topology TopologyType="{topo_type}" NumberOfElements="{num_cells}">\n'
            f'                <DataItem Format="{format_type}" DataType="Int" Dimensions="{num_cells} {dim + 1}">\n'
            f'                   {self.filename}.h5:/{step}/mesh/topology \n'
            f'                </DataItem>\n'
            f'           </Topology>\n'
            f'            <Geometry GeometryType="{geom_type}">\n'
            f'               <DataItem Format="{format_type}" Dimensions="{num_vertices} {dim}">\n'
            f'                   {self.filename}.h5:/{step}/mesh/coordinates\n'
            f'               </DataItem>\n'
            f'            </Geometry>\n'
            f'<Time Value="{time}" />\n'
            f'<Attribute\n'
            f'    ItemType="FiniteElementFunction"\n'
            f'    ElementFamily="{element}"\n'
            f'    ElementDegree="2"\n'
            f'    ElementCell="{element_cell}"\n'
            f'    Name="u"\n'
            f'    Center="Other"\n'
            f'    AttributeType="{func_type}"\n'
            f'>\n'
            f'<DataItem\n'
            f'    Dimensions="{num_cells} 30"\n'
            f'    NumberType="UInt" Format="{format_type}">\n'
            f'    {self.filename}.h5:/{step}/indices/values_indices\n'
            f'</DataItem>\n'
            f'<DataItem\n'
            f'    Dimensions="{func_shape}"\n'
            f'    NumberType="Float"\n'
            f'    Format="{format_type}"\n'
            f'>\n'
            f'    {self.filename}.h5:/{step}/values/vector\n'
            f'</DataItem>\n'
            f'</Attribute>\n'
            f'       </Grid>\n'
        )
        with open(f"{self.folder}{self.filename}.xdmf", "r") as xdmf_file:
            lines = xdmf_file.readlines()
        lines.insert(-3, xdmf)
        with open(f"{self.folder}{self.filename}.xdmf", "w") as xdmf_file:
            xdmf_file.writelines(lines)
