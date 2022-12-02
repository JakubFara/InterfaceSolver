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
