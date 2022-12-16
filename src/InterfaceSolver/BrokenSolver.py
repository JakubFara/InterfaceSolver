from dolfin import entities, entities, assemble, PETScVector, info, Cell, vertices
import petsc4py
petsc4py.init()
from petsc4py import PETSc
import numpy as np
from mpi4py import MPI
from dolfin.fem.assembling import _create_dolfin_form
from InterfaceSolver.FEniCScpp import assembler
from InterfaceSolver.InterfaceSolver import InterfaceSolver


class BrokenSolver(InterfaceSolver):

    def __init__(self, u, cell_func, interface_func, interface_boundary_func,
                 comm=None, interface_value=1, cell_val=0):
        if comm == None:
            self.comm =MPI.COMM_WORLD
        else:
            self.comm = comm
        self.mesh = u.function_space().mesh()
        self.dim = self.mesh.topology().dim()
        self.dofmap = u.function_space().dofmap()
        self.interface_boundary_dofs = self.get_interface_boundary_dofs(
            interface_boundary_func
        )
        super().__init__(
            u, cell_func, interface_func, comm, interface_value, cell_val)
        
        

    def get_interface_boundary_dofs(self, boundary_entities):
        """
        Return DOFs on the boudary. No changes will be perfomed for these DOFs.
        """
        boundary_dofs = []
        for e in entities(self.mesh, self.dim - 2):
            if boundary_entities[e] == True:
                boundary_dofs += self.entity_dofs(e, self.dofmap)

        boundary_dofs_global = self.comm.allgather(boundary_dofs)
        return set([item for sublist in boundary_dofs_global for item in sublist])
    
    def entity_dofs(self, entity, dofmap):
        dofs = dofmap.entity_dofs(self.mesh, entity.dim(), [entity.index()])
        for dim in range(entity.dim()):
            entity_indices = []
            for e in entities(entity, dim):
                entity_indices.append(e.index())
            dofs += dofmap.entity_dofs(self.mesh, dim, entity_indices)
        return [dofmap.local_to_global_index(dof) for dof in dofs]

    def assemble_interface(self, a, a1, tensor=None):
        dolfin_form = _create_dolfin_form(a, None)
        if type(self.orientation_interface) == type(None):
            self.pair_facets()
        Assembler = assembler.SingleAssembler(dolfin_form)
        w = self.get_w(Assembler)
        form_rank = Assembler.form_rank()
        if form_rank == 2:
            if type(tensor) ==type(None):
                T = PETSc.Mat().create()
                T.setSizes(tensor.shape)
                T.setType('aij')
                T.setUp()
            else:
                T = tensor
                #T.setOption(PETSc_MAT_NEW_NONZERO_ALLOCATION_ERR, PETSc_FALSE)
                T.setPreallocationNNZ(400)
        elif form_rank ==1:
            if type(tensor) ==type(None):
                T = PETScVector()
                assemble(a1, tensor=T, keep_diagonal=True)
                vec = PETSc.Vec().create(self.comm)
                vec.setSizes(T.size())
                vec.setType('mpi')
                vec.zeroEntries()
            else:
                vec = tensor
                vec.zeroEntries()
        
        for i in self.my_indices:
            lf0, lf1 = self.local_facet_interface[i, :]
            o0, o1 = self.orientation_interface[i, :]
            dc0 = list(self.dof_coordinates_interface[i, :, 0])
            dc1 = list(self.dof_coordinates_interface[i, :, 1])
            w_ = w[i]
            values = Assembler.assemble_facet(lf0, lf1, dc0, dc1, o0, o1, w_)
            #info(f'{values}')
            # info(f"{self.dofmap_interface[i, :, 0]} ::: {self.dofmap_interface[i, :, 1]}")
            # info(f'{dc0} [[[ {dc1}')
            
            rows = np.concatenate((self.dofmap_interface[i, :, 0],
                                   self.dofmap_interface[i, :, 1]), axis=None)
            
            if form_rank == 2:
                cols = np.concatenate((self.dofmap_interface[i, :, 0],
                                       self.dofmap_interface[i, :, 1]), axis=None)
                if w_ != [] or True:
                    l = len(cols)
                    values = np.resize(values, (l, l))
                    nonzero_row_ind, nonzero_col_ind = np.nonzero(values)
                    cols = cols[nonzero_col_ind]
                    rows = rows[nonzero_row_ind]
                    values = values[(nonzero_row_ind, nonzero_col_ind)]
                    for c ,r, v in zip(cols, rows, values):
                        #info(f'c = {c}')
                        if r not in self.interface_boundary_dofs:
                            T.setValues(r, c, v, addv=PETSc.InsertMode.ADD_VALUES)
                    #T.setValues(
                    #    cols, rows, values, addv=PETSc.InsertMode.ADD_VALUES)
                
            elif form_rank == 1:
                for r, v in zip(rows, values):
                    if r not in self.interface_boundary_dofs:
                        vec.setValue(
                            r, v, addv=PETSc.InsertMode.ADD_VALUES)
        self.comm.Barrier()
        if form_rank ==1:
            T = PETScVector(vec)
        return T
    
    def zero_interface_tensor(self, tensor, space, sign):
        if sign == '+':
            index = 0
        else:
            index = 1
        rows = []
        for pair in self.pairs[space].values():
            if pair[index] not in self.interface_boundary_dofs:
                rows.append(pair[index])
        #info(f'rows {rows}')
        #info(f'{self.interface_boundary_dofs}')
        tensor.zeroRows(rows, diag=0.0)
        # r = self.dofmap.ownership_range()
        # for row in rows:
        #     if r[0]<=row<r[1]:
        #         print(tensor.getRow(row))

    def zero_interface_vector(self, vector, space, sign):
        if sign == '+':
            index = 0
        else:
            index = 1
        indices = []
        zeros = []
        for pair in self.pairs[space].values():
            if pair[index] not in self.interface_boundary_dofs:
                indices.append(pair[index])
                zeros.append(0)
        #info(f'{indices}')
        vector.setValues(indices, zeros)
        # r = self.dofmap.ownership_range()
        # for i in indices:
        #     if r[0]<=i<r[1]:
        #         print(vector.getValue(i))

    def local_pair_interface_dofs(self, pairs, sub_space, space_key):
        dofmap = self.V.dofmap()
        r = dofmap.ownership_range()
        coors = self.V.tabulate_dof_coordinates().reshape((-1, self.dim))
        indices = [0, 0]
        if sub_space.num_sub_spaces() == 0:
            pairs[space_key] = {}
            for facet in self.facets:
                cell_index = facet.entities(self.dim)[0]
                c = Cell(self.mesh, cell_index)
                index = 1
                if self.cell_func[c] == self.cell_val:
                    index = 0
                sub_dofmap = sub_space.dofmap()
                vertex_coords = []
                vertex_indices = []
                for vertex in vertices(facet):
                    vertex_coords.append(list(vertex.point().array()))
                    vertex_indices.append(vertex.index())
                facet_dofs = sub_dofmap.entity_dofs(
                    self.mesh, self.mesh.topology().dim() - 1, [facet.index()]
                )
                vertex_dofs = sub_dofmap.entity_dofs(self.mesh, 0, vertex_indices)
                dofs = vertex_dofs + facet_dofs
                for dof in dofs:
                    global_dof = dofmap.local_to_global_index(dof)
                    if global_dof in self.interface_boundary_dofs:
                        continue
                    if dof<coors.shape[0]:
                        coor = coors[dof, :]
                        key = str([round(c, 5) for c in coor])
                        indices[index] += 1
                        if pairs[space_key].get(key):
                            pairs[space_key][key][index] = global_dof
                        else:
                            pairs[space_key][key] = {index: global_dof}

        else:
            for i in range(sub_space.num_sub_spaces()):
                self.local_pair_interface_dofs(pairs, sub_space.sub(i), space_key+ (i, ))