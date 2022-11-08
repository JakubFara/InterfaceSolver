import ufl
import dolfin.cpp as cpp
import dolfin
from dolfin.function.argument import TestFunction, TrialFunction
from dolfin.function.function import Function
from dolfin.fem.assembling import assemble_system
from dolfin.function.functionspace import FunctionSpace
from dolfin.fem.projection import _extract_function_space



def discontinuous_projection(
    v1, v2, dx1, dx2, V=None, bcs=None, mesh=None, function=None, 
    solver_type="lu", preconditioner_type="default", 
    form_compiler_parameters=None):
    """Return projection of given expressions *v1* and *v2* onto the finite
    element space *V*.

    *Arguments*
        v1
            a :py:class:`Function <dolfin.functions.function.Function>` or
            an :py:class:`Expression <dolfin.functions.expression.Expression>`
        v2
            a :py:class:`Function <dolfin.functions.function.Function>` or
            an :py:class:`Expression <dolfin.functions.expression.Expression>`
        dx1
            a :py:class:`Measure <dolfin.ufl.Measure>`
        dx2
            a :py:class:`Measure <dolfin.ufl.Measure>`
        bcs
            Optional argument :py:class:`list of DirichletBC
            <dolfin.fem.bcs.DirichletBC>`
        V
            Optional argument :py:class:`FunctionSpace
            <dolfin.functions.functionspace.FunctionSpace>`
        mesh
            Optional argument :py:class:`mesh <dolfin.cpp.Mesh>`.
        solver_type
            see :py:func:`solve <dolfin.fem.solving.solve>` for options.
        preconditioner_type
            see :py:func:`solve <dolfin.fem.solving.solve>` for options.
        form_compiler_parameters
            see :py:class:`Parameters <dolfin.cpp.Parameters>` for more
            information.

    *Example of usage*

        .. code-block:: python

            mesh = make_discontinuous_mesh()
            marker = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
            # ...
            v1 = Expression("sin(pi*x[0])", degree=2)
            v2 = Expression("cos(pi*x[0])", degree=2)
            dX = Measure("dx")(domain=mesh, subdomain_data=marker)
            dx1 = dX(indx1)
            dx2 = dX(indx2)
            V = FunctionSpace(mesh, "Lagrange", 1)
            Pv = project(v1, v2, dx1, dx2, V)

    """

    # Try figuring out a function space if not specified
    if V is None:
        # Create function space based on Expression element if trying
        # to project an Expression
        if isinstance(v1, dolfin.function.expression.Expression):
            if mesh is not None and isinstance(mesh, cpp.mesh.Mesh):
                V = FunctionSpace(mesh, v1.ufl_element())
            # else:
            #     cpp.dolfin_error("projection.py",
            #                      "perform projection",
            #                      "Expected a mesh when projecting an Expression")
        else:
            # Otherwise try extracting function space from expression
            V = _extract_function_space(v1, mesh)

    # Ensure we have a mesh and attach to measure
    if mesh is None:
        mesh = V.mesh()
    #dx = ufl.dx(mesh)

    # Define variational problem for projection
    w = TestFunction(V)
    Pv = TrialFunction(V)
    a1 = ufl.inner(w, Pv) * dx1
    L1 = ufl.inner(w, v1) * dx1
    
    a2 = ufl.inner(w, Pv) * dx2
    L2 = ufl.inner(w, v2) * dx2
    a = a1 + a2
    L = L1 + L2
    # Assemble linear system
    A, b = assemble_system(a, L, bcs=bcs,
                           form_compiler_parameters=form_compiler_parameters)

    # Solve linear system for projection
    if function is None:
        function = Function(V)
    cpp.la.solve(A, function.vector(), b, solver_type, preconditioner_type)

    return function
