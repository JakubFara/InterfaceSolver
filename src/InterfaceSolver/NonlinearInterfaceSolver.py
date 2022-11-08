import petsc4py
petsc4py.init()
from petsc4py import PETSc
from dolfin import (
    PETScVector, assemble, PETScMatrix, derivative, SystemAssembler, Form, 
    as_backend_type, info, DirichletBC
)
from mpi4py import MPI
from InterfaceSolver.InterfaceSolver import InterfaceSolver
from InterfaceSolver.options import opts_setup, DEFAULT_OPTIONS


class NonlinearInterfaceSolver(InterfaceSolver):

    def __init__(self, u, cell_func,
                 interface_func, comm = None,
                 interface_value=1, cell_val=0, params=None, monitor=True):
        """
        This is init
        """
        self.a0 = None
        self.a1 = None
        self.da0 = None
        self.da1 = None
        self.params = params
        self.monitor = monitor
        super().__init__(u,cell_func,
                         interface_func, comm=comm,
                         interface_value=interface_value, cell_val=cell_val)

        self.x = as_backend_type(self.u.vector())
        self.x_petsc = self.x.vec()

    def update_x(self, x):
        """
        Given a PETSc Vec x, update thex storag of our solution function u.
        """
        x.copy(self.x_petsc)
        self.x.update_ghost_values()
    
    def F(self, snes, x, F):
        self.update_x(x)  
        F0 = PETScVector()
        F1 = PETScVector()
        assemble(self.a0, tensor=F0)
        assemble(self.a1, tensor=F1)

        for bc in self.bcs_zero0:
            bc.apply(F0)

        for bc in self.bcs_zero1:
            bc.apply(F1)

        x.assemble()
        F.zeroEntries()
        self.assemble_interface(
            self.a_interface, self.a1+self.a0, tensor=F, finalize=True)

        for space, func, sign in self.dirichlet_interface:
            self.assemble_dirichlet_interface_vector(
                F, self.x.vec(), space, func, sign
            )
        
        F_  = PETScVector(F)
        F_.update_ghost_values()
        F_.apply('add')
        F_.axpy(1, F1)
        F_.axpy(1, F0)

        for bc in self.bcs0:
            bc.apply(F_, self.x)
            
        for bc in self.bcs1:
            bc.apply(F_,self.x)
            
        F_.apply('add')
        F_.update_ghost_values()
                          
    def J(self, snes, x, J, P):
        self.update_x(x)
        A0 = PETScMatrix()
        A1 = PETScMatrix()
        assemble(self.da0 , tensor = A0, keep_diagonal=True)
        assemble(self.da1 , tensor = A1, keep_diagonal=True)
        
        for bc in self.bcs_zero0:
            bc.zero(A0)
            
        for bc in self.bcs_zero1:
            bc.zero(A1)

        self.assemble_interface(
            self.da_interface, self.da1, tensor = J, finalize = True)

        for space, func, sign in self.dirichlet_interface:
            self.assemble_dirichlet_interface_tensor(
                J, self.x.vec(), space, func, sign
        )
        J.assemble()
        J_ = PETScMatrix(J)
        
        J_.axpy(1, A0, False)
        J_.axpy(1, A1, False)

        for bc in self.bcs0:
            bc.apply(J_)
            
        for bc in self.bcs1:
            bc.apply(J_)
        
        J.assemble()
        return True

    def solve(self, a0, a1, a_interface,
            bcs0=None, bcs1=None, bcs_zero0 = None, bcs_zero1=None,
            force_equality=None, dirichlet_interface=None,
            *args, **kwargs):

        self.a0 = a0
        self.a1 = a1
        
        self.a_interface = a_interface

        self.da0 = derivative(a0, self.u)
        self.da1 = derivative(a1, self.u)
        self.da_interface = derivative(a_interface,self.u)

        self.bcs0 = bcs0
        self.bcs1 = bcs1
        if bcs_zero0 == None:
            self.bcs_zero0 = []
        else:
            self.bcs_zero0 = [*bcs_zero0]
            
        if bcs_zero1 == None:
            self.bcs_zero1 = []
        else:
            self.bcs_zero1 = [*bcs_zero1]

        if force_equality == None:
            self.force_equality = ()
        else:
            self.force_equality = force_equality
        
        if dirichlet_interface == None:
            self.dirichlet_interface = []
        else:
            self.dirichlet_interface = dirichlet_interface
        
        # for space, _, sign in self.dirichlet_interface:
        #     V = self.V
        #     for i in space:
        #         V = V.sub(i)
        #     if sign == '+':
        #         self.bcs_zero1.append(
        #             DirichletBC(
        #                 V, 0.0, self.interface_func, self.interface_value)
        #             )
        #     else:
        #         self.bcs_zero0.append(
        #             DirichletBC(
        #                 V, 0.0, self.interface_func, self.interface_value)
        #             )
        
        self.form_compiler_parameters = {"optimize": True}
        if "form_compiler_parameters" in kwargs:
            self.form_compiler_parameters = kwargs["form_compiler_parameters"]
        self.assembler = SystemAssembler(self.da0 + self.da1, self.a0 + self.a1,
                                         self.bcs0, form_compiler_parameters=
                                         self.form_compiler_parameters)
        self.comm.Barrier()

        self.A_dolfin = PETScMatrix()
        self.assembler.init_global_tensor(self.A_dolfin, Form(self.da0 + self.da1))
        self.A_petsc = self.A_dolfin.mat()

        self.xx=self.A_petsc.createVecRight()
        self.xx.axpy(1.0,self.x_petsc)
        self.b_petsc = self.A_petsc.createVecLeft()

        self.set_solver(params=self.params, monitor=self.monitor)
        self.snes.setFromOptions()
        self.snes.setUp()
        self.snes.setSolution(self.xx)
        self.snes.solve(None,self.xx)

        for subspace, sign in  self.force_equality:
            self.assign_interface(self.x.vec(), self.x.vec(), subspace, sign)
        self.x.update_ghost_values()


    def set_solver(self, params=None, monitor=True):
        self.snes = PETSc.SNES().create(self.comm)
        self.ksp = self.snes.getKSP()
        self.snes.setFunction(self.F, self.b_petsc)
        self.snes.setJacobian(self.J, self.A_petsc)
        self.snes.setSolution(self.xx)
        self.snes.computeFunction( self.xx,  self.b_petsc)
        self.snes.computeJacobian( self.xx, self.A_petsc)

        if monitor==True:
            self.snes.setMonitor(SNESMonitor())
        if not params:
            params = DEFAULT_OPTIONS

        opts_setup(params)
        self.snes.setFromOptions()
        self.ksp.setFromOptions()


class SNESMonitor():
    def __init__(self):
        self.init = True
        self.line= "_"*50
        self.print = PETSc.Sys.Print

    def __call__(self, snes, its, rnorm, *args, **kwargs):
        if self.init==True:
            s=('%6s' % "it") + (' %5s '% '|') + (' %10s ' % "rnorm")
            self.print(s)
            self.print(self.line)

            self.init =False
        s = ('%6d' % its) + (' %5s '% '|') + (' %12.2e' % rnorm)
        self.print(s)
        MPI.COMM_WORLD.Barrier()
        xnorm=snes.vec_sol.norm(PETSc.NormType.NORM_2)
        ynorm=snes.vec_upd.norm(PETSc.NormType.NORM_2)
        iterating = (snes.callConvergenceTest(its, xnorm, ynorm, snes.norm) == 0)
        if not iterating :
            self.init=True
            s = (
                "Convergence reason: {0}    "
                "iterations = {1}".format(snes.reason,its)
            )
            self.print(s)
            self.print(self.line)
            
    def info(self,s):
        if MPI.COMM_WORLD.Get_rank() == 0:
            info(s)

class KSPMonitor(object):
    def __init__(self, name='KSP'):
        self.name=name
        
    def __call__(self, ksp, its, rnorm, *args, **kwargs):
        if (its==0):
            self.rnorm0 = rnorm
            PETSc.Sys.Print(self.name)
        else:
            if ksp.iterating :
                if (its==1) :
                    PETSc.Sys.Print((
                        ('%6s' % "it") + (' %12s' % "l2-norm") + 
                        (' %12s' % "rel l2-norm") + (' %12s' % "l1-norm") + 
                        (' %12s' % "inf-norm")+ (' %12s' % "energy-norm")
                    ))
                r=ksp.buildResidual()
                rn1 = r.norm(PETSc.NormType.NORM_1)
                rn2 = r.norm(PETSc.NormType.NORM_2)
                rni = r.norm(PETSc.NormType.NORM_INFINITY)
                rne = r.dot(ksp.vec_sol)
                PETSc.Sys.Print(
                    ('%6d' % its) + (' %12.2e' % rnorm) + 
                    (' %12.2e' % (rnorm/self.rnorm0))+ (' %12.2e' % (rn1))+ 
                    (' %12.2e' % (rni))+ (' %12.2e' % (rne))
                )
            else:
                PETSc.Sys.Print(("Result: {0}".format(ksp.reason)))
                self.solver.ksp_its=its
                
