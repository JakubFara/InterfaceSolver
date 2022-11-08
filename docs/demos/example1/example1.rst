**************************
Build a Discontunuous mesh
**************************

We will create a discontinuous mesh. In this example we will create a square
mesh which will be suitable for the following examples.

Implementation
##############
At first we import function and dolfin
::

    import dolfin
    from InterfaceSolver import make_discontinuous_mesh
    
Then we will reate a mesh which we would like to split. In this example we will
use square mesh generated from dolfin
::

    mesh = dolfin.UnitSquareMesh(20, 20, "crossed")

We will mark the upper and bottom parts of the domain into class 
`dolfin.MeshFunction`
::

    marker = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    for c in dolfin.cells(mesh):
        marker[c] = c.midpoint().y() < 0.5

And we will store this mesh into file as "directory/name.xml"
::
    
    make_discontinuous_mesh(
        mesh, marker, 1, 0, save=True, directory=directory, name=name)
