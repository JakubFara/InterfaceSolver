from dolfin import (
    PETScMatrix, PETScVector, assemble, solve
)
import petsc4py
petsc4py.init()
from InterfaceSolver.InterfaceSolver import InterfaceSolver

class LinearInterfaceSolver(InterfaceSolver):
    def __init__(self, u, cell_func,
                 interface_func, comm=None,
                 interface_value=1, cell_val=0):

        super().__init__(u, cell_func,
                         interface_func,comm=comm,
                         interface_value=interface_value, cell_val=cell_val)

    def solve(self, a0, a1, a_interface,
            l0=None, l1=None, l_interface=None,
            bcs0=None, bcs1=None, bcs_zero0=None, bcs_zero1=None):
        if bcs0 ==None:
            bcs0 = []
        if bcs1 == None:
            bcs1 = []
        if bcs_zero0 ==None:
            bcs_zero0 = []
        if bcs_zero1 == None:
            bcs_zero1 = []
        
        A0 = PETScMatrix()
        assemble(a0, tensor= A0, keep_diagonal = True)
            
        A1 = PETScMatrix()
        assemble(a1, tensor= A1, keep_diagonal = True)

        A_interface = PETScMatrix()
        assemble(a0, tensor=A_interface, keep_diagonal = True)
        A_interface.zero()

        if l0 == None:
            v0 = A0.mat().getVecLeft()
            v0.zeroEntries()
            L0 = PETScVector(v0)
        else:
            L0  = PETScVector()
            assemble(l0, tensor=L0)
            
        if l1 == None:
            v1 = A1.mat().getVecLeft()
            v1.zeroEntries()
            L1 = PETScVector(v1)
        else:
            L1  = PETScVector()
            assemble(l1, tensor=L1)
            
        for bc in bcs_zero0:
            bc.zero(A0)
            bc.apply(L0)

        for bc in bcs_zero1:
            bc.zero(A1)
            bc.apply(L1)

        A1.apply('add')
        #A_interface = PETScMatrix()
        self.assemble_interface(a_interface, a1, tensor=A_interface.mat(), finalize=True)
        A0.apply('add')
        A0.mat().assemble()
        A0.axpy(1, A1, False)
        A_interface.mat().assemble()
        A0.axpy(1, A_interface, False)

        L0.vec().axpy(1,L1.vec())

        for bc in bcs0:
            bc.apply(A0)
            bc.apply(L0)
            
        for bc in bcs1:
            bc.apply(A0)
            bc.apply(L0)

        if l_interface != None:
            self.assemble_interface(l_interface, l0, tensor=L0.vec(), finalize=True)
            #L0.vec().axpy(1, l_interface.vec())
        solve(A0, self.u.vector(), L0)
