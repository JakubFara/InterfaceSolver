from dolfin import vertices
from InterfaceSolver.FEniCScpp import FEniCScpp,assembler
from mpi4py import MPI
import petsc4py
petsc4py.init()
from petsc4py import PETSc
import numpy as np
from dolfin.fem.assembling import _create_dolfin_form
from dolfin import facets, Cell, entities, assemble, PETScVector, info


PETSc_MAT_NEW_NONZERO_ALLOCATION_ERR = 19
PETSc_FALSE = 0
PETSc_TRUE = 1
EPSILON = 0.000001

class InterfaceSolver(object):
    def __init__(self,u,cell_func,
                 interface_func,comm=None,
                 interface_value=1, cell_val =0):
        if comm == None:
            self.comm =MPI.COMM_WORLD
        else:
            self.comm = comm
        self.rank = self.comm.Get_rank()
        self.comm_size = self.comm.Get_size()
        self.u = u
        self.V = self.u.function_space()
        self.mesh = self.V.mesh()
        self.cell_func = cell_func
        self.interface_func = interface_func
        self.cell_val = cell_val
        self.interface_value = interface_value
        self.dofmap = self.V.dofmap()
        self.dofs_in_cell = FEniCScpp.num_dofs_in_cell(self.dofmap)
        self.dim = self.mesh.topology().dim()

        self.orientation_interface = None
        self.local_facet_interface = None
        self.dof_coordinates_interface = None
        self.dofmap_interface = None
        self.cells = None
        self.pair_facets()
        self.pairs = self.get_pairs()

    def pair_facets(self):
        """
        This function colect data about cells on the interface. The result is
        written into the following structures.

            self.my_indices
                list: indices owned by this process.

            self.orientation_interface
                np.array: orientation off cells on the interface

            self.local_facet_interface
                np.array: facet located on the interface

            self.dof_coordinates_interface 
                np.array: coordinates of the interface cells
                                    
            self.dofmap_interface 
                np.array: dofs on the interface
        
            self.cells
                np.array: indices of cells

        """
        facets_list = []
        for f in entities(self.mesh, self.dim - 1):
            if len(f.entities(2)) != 1 or self.interface_func[f] != self.interface_value:
                continue
            facets_list.append(f)

        self.facets = facets_list
        size_local = len(facets_list)
        
        size = self.comm.allreduce(sendobj=size_local, op=MPI.MAX)
        
        X = np.zeros((size,  self.dim))
        X.fill(np.nan)
        orientation = np.zeros((size, 2), dtype='int32')
        local_facet = np.zeros((size, 2), dtype='int32')
        dof_coordinates = np.zeros((size, self.dim*(self.dim + 1), 2), dtype='double')
        dofmap = np.zeros((size, self.dofs_in_cell, 2), dtype='int32')
        cells = np.full((size, 2), -1, dtype='int32')

        my_indices = []
        i = 0
        for f in facets_list:
            cell_index = f.entities(self.dim)[0]
            c = Cell(self.mesh,cell_index)
            index = 1
            if self.cell_func[c] == self.cell_val:
                index = 0
            lf = FEniCScpp.local_facet(f, c)
            o = FEniCScpp.orientation(c, lf)
            dc = c.get_coordinate_dofs()
            dm = np.array([self.dofmap.local_to_global_index(dof) for
                           dof in self.dofmap.cell_dofs(cell_index) ])
            mid = FEniCScpp.middle_point(f)
            if self.dim == 2:
                ids = np.argwhere(
                    (X[:,0]-mid.x())**2+(X[:,1]-mid.y())**2 < EPSILON
                )
            elif self.dim == 3:
                ids = np.argwhere(
                    (X[:, 0]-mid.x())**2 + (X[:, 1]-mid.y())**2 (X[:, 2]-mid.z())**2 < EPSILON
                )
            if ids.size ==0:
                X[i,0] = mid.x()
                orientation[i, index] = o
                local_facet[i, index] = lf
                dof_coordinates[i,:,index] = dc
                dofmap[i, :, index] = dm
                X[i,1] = mid.y()
                cells[i, index] = cell_index
                if index ==0:
                    my_indices.append(i) # assemble only facets where I own cell with index 0
                i+=1
            else:
                pos = ids[0][0]
                orientation[pos, index] = o
                local_facet[pos, index] = lf
                dof_coordinates[pos, :, index] = dc
                dofmap[pos, :, index] = dm
                cells[pos, index] = cell_index
                if index ==0:
                    my_indices.append(pos) # assemble only facets where I own cell with index 0
                    
        p = self.comm_size
        X_global = np.zeros((size*p, 2))
        orientation_global = np.zeros((size*p, 2),dtype = 'int32')
        local_facet_global = np.zeros((size*p, 2),dtype = 'int32')
        dof_coordinates_global = np.zeros((size*p, self.dim*(self.dim + 1), 2))
        dofmap_global = np.zeros((size*p, self.dofs_in_cell, 2), dtype = 'int32')
        
        self.comm.Allgather([X, MPI.DOUBLE],[X_global, MPI.DOUBLE])
        self.comm.Allgather(
            [orientation, MPI.DOUBLE],
            [orientation_global, MPI.DOUBLE]
        )
        self.comm.Allgather(
            [local_facet, MPI.DOUBLE],
            [local_facet_global, MPI.DOUBLE]
        )
        self.comm.Allgather(
            [dof_coordinates, MPI.DOUBLE],
            [dof_coordinates_global, MPI.DOUBLE]
        )
        self.comm.Allgather(
            [dofmap, MPI.DOUBLE],
            [dofmap_global, MPI.DOUBLE]
        )
        
        indices = [i for i in range(size*(p)) if (i<size*self.rank or i>=size*(self.rank+1))]
        X_global = X_global[indices,:]
        self.positions  = np.full((size), -1, dtype='int32')
        for i in my_indices:
            mid_x = X[i, :]
            if self.dim ==3:
                ids = np.argwhere(
                    ((X_global[:, 0]-mid_x[0])**2 +
                    (X_global[:, 1]-mid_x[1])**2 +
                    (X_global[:, 2]-mid_x[2])**2)  <EPSILON
                )
            elif self.dim == 2:
                ids = np.argwhere(
                    ((X_global[:, 0]-mid_x[0])**2 +
                    (X_global[:, 1]-mid_x[1])**2)  <EPSILON
                ) 
            if ids.size != 0:
                k = indices[ids[0][0]]
                self.positions[i] = k
                orientation[i, 1] = orientation_global[k, 1]
                local_facet[i, 1] = local_facet_global[k, 1]
                dof_coordinates[i, :, 1] = dof_coordinates_global[k, :, 1]
                dofmap[i, :, 1] = dofmap_global[k, :, 1]

        self.my_indices = my_indices
        self.orientation_interface = orientation
        self.local_facet_interface = local_facet
        self.dof_coordinates_interface = dof_coordinates
        self.dofmap_interface = dofmap
        self.cells = cells

    def local_pair_interface_dofs(self, pairs, sub_space, space_key):
        dofmap = self.V.dofmap()
        r = dofmap.ownership_range()
        coors = self.V.tabulate_dof_coordinates().reshape((-1, self.dim))
        indices = [0, 0]
        if sub_space.num_sub_spaces() == 0:
            pairs[space_key] = {}
            for facet in self.facets:
                cell_index = facet.entities(self.dim)[0]
                c = Cell(self.mesh,cell_index)
                index = 1
                if self.cell_func[c] == self.cell_val:
                    index = 0
                sub_dofmap = sub_space.dofmap()
                vertex_coords = []
                vertex_indices = []
                for vertex in vertices(facet):
                    vertex_coords.append(list(vertex.point().array()))
                    vertex_indices.append(vertex.index())
                facet_dofs = sub_dofmap.entity_dofs(self.mesh, self.mesh.topology().dim() - 1, [facet.index()])
                vertex_dofs = sub_dofmap.entity_dofs(self.mesh, 0, vertex_indices)
                dofs = vertex_dofs + facet_dofs
                for dof in dofs:
                    global_dof = dofmap.local_to_global_index(dof)
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
        

    def get_pairs(self):
        pairs = {}
        self.local_pair_interface_dofs(pairs, self.V, ())
        global_pairs = self.comm.allgather(pairs)
        for space_key, sub_pairs in pairs.items():
            for key, pair in sub_pairs.items():
                
                if len(pair) == 1:
                    missing_index = 1-[*pair.keys()][0]
                    found = False
                    for global_pair in global_pairs:
                        if global_pair[space_key].get(key) and (global_pair[space_key][key].get(missing_index) != None):
                            sub_pairs[key][missing_index] = global_pair[space_key][key][missing_index]
                            found = True
                            break

                    if not found:
                        info('NOT FOUND PAIR !!!!!!!!!!!!!!!')
                        print(space_key, key, missing_index)

        return pairs

    def assign_interface(self, x1, x2, subspace, sign):
        if sign == '+':
            index = 0
        else:
            index = 1
        dofmap = self.V.dofmap()
        r = dofmap.ownership_range()
        values = {}
        dofs2 = [pair[1-index] for pair in self.pairs[subspace].values()]
        global_dofs2 = self.comm.allgather(dofs2)
        for dofs in global_dofs2:
            for dof in dofs:
                if r[0]<=dof<r[1]:
                    val = x2.getValue(dof)
                    values[dof] = val

        global_values = self.comm.allgather(values)
        for pair in self.pairs[subspace].values():
            dof1 = pair[index]
            dof2 = pair[1-index]
            for global_val in global_values:
                val = global_val.get(dof2)
                if val and r[0]<=dof1<r[1]:
                    x1.setValue(dof1, val)
                    break

    def get_w(self, Assembler):
        """
        Get data to Constants, Functions ... in UFL expression
        """
        #get local coefficients
        W0 = []
        W1 = []
        for i in range(self.cells.shape[0]):
            w0 = []
            w1 = []            
            c_index = self.cells[i,0]
            if c_index != -1:
                c = Cell(self.mesh, c_index)
                local_facet = self.local_facet_interface[i, 0]
                dc = list(self.dof_coordinates_interface[i, :, 0])
                w0 = Assembler.w(c, local_facet,dc)
            c_index = self.cells[i, 1]
            if c_index != -1:
                c = Cell(self.mesh,c_index)
                local_facet = self.local_facet_interface[i, 1]
                dc = list(self.dof_coordinates_interface[i, :, 1])
                w1 = Assembler.w(c, local_facet,dc)
            W0.append(w0)
            W1.append(w1)
        #W0_global = [i for p in self.comm.allgather(W0) for i in p]
        W1_global = [i for p in self.comm.allgather(W1) for i in p]
        for i in self.my_indices:
            if self.cells[i, 1] == -1:
                W1[i] = W1_global[self.positions[i]]
            for k in range(len(W0[i])):
                W0[i][k] = W0[i][k] +W1[i][k]
        return W0


    def assemble_dirichlet_interface_tensor(
            self, tensor, x, space, func, sign:str
        ):
        T = tensor
        if sign == '+':
            index = 0
        else:
            index = 1
        if func == None:
            for pair in self.pairs[space].values():
                T.setValues(
                    [pair[index], pair[1 - index]],
                    [pair[index], pair[1 - index]],
                    [1.0, -1.0, 0.0, 0.0],
                    addv=PETSc.InsertMode.ADD_VALUES
                )
        else:
            values = self.get_nodes_values(x, index)
            spaces = list(self.pairs.keys())
            
            nodes = set()
            for s  in spaces:
                nodes = nodes.union(set(self.pairs[s].keys()))
            for node in nodes:
                # do it only for your rank
                if self.pairs[space].get(node) == None:
                    continue

                dofs = ([],[])
                for s in spaces:
                    if not self.pairs[space].get(node):
                        # if the space has not this node
                        continue

                    pair = self.pairs[s].get(node)
                    if pair:
                        dofs[index].append(pair[index])
                        dofs[1-index].append(pair[1-index])
                    else:
                        dofs[index].append(None)
                        dofs[1-index].append(None)
                
                dofs = dofs[0] + dofs[1]
                #info(f'{dofs} {self.pairs[space][node][index]}')
                vals = func.jacobian(
                    node, values[node][0], values[node][1]
                )
                vals = [ vals[i] for i, dof in enumerate(dofs) if dof != None]
                dofs = [ dof for dof in dofs if dof != None]
                
                T.setValues(
                    [self.pairs[space][node][index]],
                    dofs,
                    vals,
                    addv=PETSc.InsertMode.ADD_VALUES
                )
        #T.setOption(PETSc_MAT_NEW_NONZERO_ALLOCATION_ERR, PETSc_TRUE)
        T.setUp()
        #T.assemble()

    def get_nodes_values(self, x, index):
        values = {}
        dofmap = self.V.dofmap()
        r = dofmap.ownership_range()

        spaces = list(self.pairs.keys())
        nodes = set()
        for s  in spaces:
            nodes = nodes.union(set(self.pairs[s].keys()))

        global_pairs = self.comm.allgather(self.pairs)
        # initialize the dicionary
        for node in nodes:
            values[node] = {}
            for i in [index, 1-index]:
                values[node][i] = {}
        # find the local values
        for pairs in global_pairs:
            for s, nodes in pairs.items():
                for node, pair in nodes.items():
                    if r[0]<=pair[index]<r[1]:
                        values[node][index][s] = x.getValue(pair[index])
                    if r[0]<=pair[1-index]<r[1]:
                        values[node][1-index][s] = x.getValue(pair[1 - index])
        for coor in values.keys():
            if values[coor] == {}:
                del values[coor]
                continue
            for i in [index, 1-index]:
                if values[coor][i] == {}:
                    del values[coor][i]
        # share local values with the rest of the processes
        gloval_values_per_proc = self.comm.allgather(values)
        
        # union of the dictionaries
        global_values = {}
        for local_values in gloval_values_per_proc:
            for coor, node in local_values.items():
                if global_values.get(coor) == None:
                    global_values[coor] = node
                else:
                    for i, v in node.items():
                        if global_values[coor].get(i) == None:
                            global_values[coor][i] = v
                        else:
                            global_values[coor][i] = {**v, **global_values[coor][i]}
        
        return global_values
            

    def assemble_dirichlet_interface_vector(
        self, vector, x, space, func, sign:str):
        if sign == '+':
            index = 0
        else:
            index = 1
        dofmap = self.V.dofmap()
        r = dofmap.ownership_range()
        values = {}
        dofs2 = [pair[1-index] for pair in self.pairs[space].values()]
        global_dofs2 = self.comm.allgather(dofs2)
        for dofs in global_dofs2:
            for dof in dofs:
                if r[0]<=dof<r[1]:
                    val = x.getValue(dof)
                    values[dof] = val

        global_values = self.comm.allgather(values)
        if func == None:
            for pair in self.pairs[space].values():
                dof1 = pair[index]
                dof2 = pair[1-index]
                for global_val in global_values:
                    val2 = global_val.get(dof2)
                    if val2 and r[0]<=dof1<r[1]:
                        val1 = x.getValue(dof1)
                        vector.setValue(
                            dof1, val1-val2, addv=PETSc.InsertMode.ADD_VALUES)
                        break
        else:
            values = self.get_nodes_values(x, 0)
            for coor, node in values.items():
                if self.pairs[space].get(coor)==None:
                    continue
                dof = self.pairs[space][coor][index]
                if r[0]<=dof<r[1]:
                    residual = func.residual(coor, node[0], node[1])
                    vector.setValue(
                            dof, residual, addv=PETSc.InsertMode.ADD_VALUES)

    def assemble_interface(self, a, a1, tensor=None, finalize=True):   
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
            lf0,lf1 = self.local_facet_interface[i, :]
            o0,o1 = self.orientation_interface[i, :]
            dc0 = list(self.dof_coordinates_interface[i, :, 0])
            dc1 = list(self.dof_coordinates_interface[i, :, 1])
            w_ = w[i]
            values = Assembler.assemble_facet(lf0, lf1, dc0, dc1, o0, o1, w_)
            
            cols = np.concatenate((self.dofmap_interface[i, :, 0],
                                   self.dofmap_interface[i, :, 1]), axis=None)
            
            if form_rank == 2:
                rows = np.concatenate((self.dofmap_interface[i, :, 0],
                                       self.dofmap_interface[i, :, 1]), axis=None)
                if w_ != []:
                    l = len(cols)
                    values = np.resize(values, (l, l))
                    nonzero_col_ind, nonzero_row_ind = np.nonzero(values)
                    cols = cols[nonzero_col_ind]
                    rows = rows[nonzero_row_ind]
                    values = values[(nonzero_col_ind, nonzero_row_ind)]
                    for c ,r, v in zip(cols, rows, values): 
                        T.setValues(c, r, v, addv=PETSc.InsertMode.ADD_VALUES)
                    #T.setValues(
                    #    cols, rows, values, addv=PETSc.InsertMode.ADD_VALUES)
                
            elif form_rank ==1:
                vec.setValues(
                    cols, values, addv=PETSc.InsertMode.ADD_VALUES)
        self.comm.Barrier()
        if form_rank ==1:
            T = PETScVector(vec)
        return T