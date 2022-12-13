import dolfin as df
import numpy as np


def connect_subdomains(
    submesh1, submesh2, save = False, directory = '', name = 'mesh'):
    """
    This function connect 2 submeshes with common part of boundary into 'mesh'.

    *Arguments*
        submesh1
            a :py:class:`dolfin.mesh <dolfin.cpp.Mesh>` or
            a :py:class:`dolfin.SubMesh <dolfin.cpp.mesh.SubMesh>`
        submesh2
            a :py:class:`dolfin.mesh <dolfin.cpp.Mesh>` or
            a :py:class:`dolfin.SubMesh <dolfin.cpp.mesh.SubMesh>`
        save
            `bool`
        directory
            `str` directory to store the mesh
        name
            `str` name of the mesh (with no prefix)
    """
    coords1 = submesh1.coordinates()[:]
    cls1 = submesh1.cells()[:]
    coords2 = submesh2.coordinates()[:]
    cls2 = submesh2.cells()[:] + (coords1.shape[0])
    cls = np.concatenate((cls1, cls2))
    coords = np.concatenate((coords1, coords2))
    
    mesh = dolfin_mesh(coords, cls)
    if save == True:
        mesh_path = directory + name + '.xml'
        mesh_file = df.File(mesh_path)
        mesh_file << mesh
    return mesh

def dolfin_mesh(coords, cls):
    """
    Create `mesh <dolfin.cpp.Mesh>` object from list of coordinates and list of
    cells. This function works only in serial.

    *Arguments*
        coords
            `list`
        cls
            `list`
    """
    dim = cls.shape[1] - 1
    n_coords = coords.shape[0] 
    n_cls = cls.shape[0]
    if dim == 2:
        cell_type = 'triangle'
    elif dim == 3:
        cell_type = 'tetrahedron'
    
    editor = df.MeshEditor()
    mesh = df.Mesh()
    editor.open(mesh, cell_type, dim, dim)  # top. and geom. dimension are both dim
    editor.init_vertices(n_coords)  # number of vertices
    editor.init_cells(n_cls)  # number of cells
    for i in range(n_coords):
        editor.add_vertex(i, coords[i][: dim])
    for i in range(n_cls):
        editor.add_cell(i, cls[i])
    editor.close()
    return mesh

def make_discontinuous_mesh(
    mesh, marker, val0, val1, save=True, directory='', name ='mesh'):
    """
    *Arguments*
        mesh
            a :py:class:`dolfin.Mesh <dolfin.cpp.Mesh>`
        marker
            a :py:class:`dolfin.MeshFunction`
        val0
            `int`
        val1
            `int`
        save
            optional argument `bool`
        directory
            optional argument `str`
        name
            optional argument `str`

    """
    submesh0 = df.SubMesh(mesh, marker, val0)
    submesh1 = df.SubMesh(mesh, marker, val1)
    new_mesh = connect_subdomains(
        submesh0, submesh1, save=save, directory=directory, name=name
    )
    return new_mesh


def boundary_edge(edge: df.MeshEntity, dim: int, interface: df.MeshFunction,
                  val: int):
    """
    """
    num_edges = 0
    for e in df.entities(edge, dim -1):
        if interface[e] == val:
            num_edges += 1
    if num_edges == 1:
        return True
    return False


def color_neighbour(cell: df.MeshEntity, interface: df.MeshFunction,
                    interface_cells: df.MeshFunction,
                    boundary_entities: df.MeshFunction, val: int, dim: int):
    """
    """
    interface_cells[cell] = 2
    for edge in df.entities(cell, dim - 1):
        boundary = False
        for v in df.entities(edge, dim - 2):
            if boundary_entities[v] == 1:
                boundary = True
        if interface[edge] == val or boundary:
            continue
        for c in df.entities(edge, dim):
            if interface_cells[c] == 1:
                color_neighbour(
                    c, interface, interface_cells, boundary_entities, val, dim
                )


def interface_vertex(v, interface, val: int, dim: int):
    for e in df.entities(v, dim - 1):
        if interface[e] == val:
            return True
    return False


def make_broken_mesh(mesh: df.Mesh, interface: df.MeshFunction, val: int,
                     directory='./', name='broken_mesh'):
    """
    """
    dim = mesh.topology().dim()
    boundary_entities = df.MeshFunction('size_t', mesh, dim - 2, 0)
    interface_cells = df.MeshFunction('size_t', mesh, dim, 0)
    # mark entities on the boundary of the interface
    for edge in df.entities(mesh, dim - 1):
        if interface[edge] != val:
            continue
        for e in df.entities(edge, dim - 2):
            if boundary_edge(e, dim, interface, val):
                boundary_entities[e] = 1

    # mark cells near interface
    for edge in df.entities(mesh, dim - 1):
        if interface[edge] != val:
            continue
        for e in df.entities(edge, dim - 2):
            if boundary_entities[e] != 1:
                for c in df.entities(e, dim):
                    interface_cells[c] = 1

    # find initial cell
    initial_cell = None
    for edge in df.entities(mesh, dim - 1):
        if interface[edge] == val:
            for c in df.entities(edge, dim):
                initial_cell = c
                break
            break
    
    color_neighbour(
        initial_cell, interface, interface_cells, boundary_entities, val, dim
    )
    subdomain_file = df.File("subdomains.pvd")
    subdomain_file << (interface_cells)
    coords = mesh.coordinates()[:]
    cls = mesh.cells()[:]
    new_index_last = (cls[:, 0]).size
    changed_indices = {}
    for c in  df.entities(mesh, dim):
        if interface_cells[c] == 2:
            cell_index = c.global_index()
            for v in df.entities(c, dim - 2):
                if boundary_entities[v] != 1 and interface_vertex(v, interface, val, dim):
                    index = v.global_index()
                    if not changed_indices.get(index):
                        changed_indices[index] = new_index_last
                        new_index = new_index_last
                    else:
                        new_index = changed_indices[index]
                    j = np.where(cls[cell_index, :] == index)[0]
                    if j.size == 1:
                        j = j[0]
                    else:
                        print('ERRORRRR')
                    cls[cell_index, j] = new_index
                    new_index_last += 1

    coords_new = np.zeros((new_index_last, dim))
    coords_new[: coords.shape[0], : coords.shape[1]] = coords
    for i, new_i in changed_indices.items():
        coords_new[new_i, :] = coords[i, :]

    mesh = dolfin_mesh(coords_new, cls)

    mesh_file = df.File(directory + name + '.xml')
    mesh_file << mesh


def interface(mesh, func, val=1, eps=0.0001):
    """
    Labeling of facets for which `func` < `eps`.

    *Arguments*
        mesh
            a `dolfin.mesh`
        func
            function with tho arguments x, y, which returns float
        val
            `int` by which will be labeled the interface facets 
    """
    dim = mesh.topology().dim()
    interface = df.MeshFunction("size_t", mesh, dim - 1, 0)

    for f in df.entities(mesh, dim - 1):
        mid = f.midpoint()
        if dim == 2:
            func_val = func(mid.x(), mid.y())
        elif dim == 3:
            func_val = func(mid.x(), mid.y(), mid.z())
        if abs(func_val) < eps:
            interface[f] = val
    return interface
