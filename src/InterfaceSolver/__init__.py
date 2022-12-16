from InterfaceSolver.DiscontinuousMesh import (
    connect_subdomains, make_discontinuous_mesh, interface, make_broken_mesh
)
from InterfaceSolver.DiscontinuousProjection import discontinuous_projection
from InterfaceSolver.LinearInterfaceSolver import LinearInterfaceSolver
from InterfaceSolver.NonlinearInterfaceSolver import (
    NonlinearInterfaceSolver, SNESMonitor, KSPMonitor
)
from InterfaceSolver.NonlinearBrokenSolver import NonlinearBrokenSolver