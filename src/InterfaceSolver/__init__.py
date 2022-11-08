from InterfaceSolver.DiscontinuousMesh import (
    connect_subdomains, make_discontinuous_mesh, interface
)
from InterfaceSolver.DiscontinuousProjection import discontinuous_projection
from InterfaceSolver.LinearInterfaceSolver import LinearInterfaceSolver
from InterfaceSolver.NonlinearInterfaceSolver import (
    NonlinearInterfaceSolver, SNESMonitor, KSPMonitor
)