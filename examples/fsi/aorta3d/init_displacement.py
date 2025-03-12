import dolfin as df
from ufl import atan_2
from scipy.interpolate import CubicSpline, CubicHermiteSpline


class UInit(df.UserExpression):
    def __init__(self, r_max, scale=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r_max = r_max
        self.set_parameters()
        self.scale = scale

    def eval(self, value, x):
        exp_shape = self.exp_shape(x)
        phi = atan_2(-x[1], -x[0])
        if x[2] < 0:
            D = self.D_in
        else:
            D = self.D_out
        # r_scale = min(df.sqrt(x[0]**2 + x[1]**2) / (self.D_in / 2), 1) 
        if self.scale == None:
            r_scale = df.sqrt(x[0]**2 + x[1]**2) / (self.D_in / 2) 
        elif isinstance(self.scale, float) or isinstance(self.scale, int):
            r_scale = self.scale
        else:
            df.info(f"WARNING scale in UInit is not `None`, `float` or `int`, but {type(self.scale)}")
            
        value[0] = (
            r_scale * (D / 2 + (self.R + self.r - D / 2) * exp_shape) * df.cos(phi)
            - r_scale * self.lam * self.r * df.exp(-8000 * x[2]**2) * df.cos(((self.R + self.r) / self.r) * phi)
            - x[0]
        )
        value[1] = (
            r_scale * (D / 2 + (self.R + self.r - D / 2) * exp_shape) * df.sin(phi)
            - r_scale * self.lam * self.r * df.exp(-8000 * x[2]**2) * df.sin(((self.R + self.r) / self.r) * phi)
            - x[1]
        )
        value[2] = 0 
    
    def exp_shape(self, x):
        if x[2] < 0:
            return df.exp(- self.popt_in[0] * x[2]**2 - self.popt_in[1] * x[2]**4)
        else:
            return df.exp(- self.popt_out[0] * x[2]**2 - self.popt_out[1] * x[2]**4)

    def set_parameters(self):
        self.L = 0.044     
        self.lam = 0.5 
        self.D_in = 0.024 
        self.D_out = 0.026 
        if self.r_max == 0.016:
           self.r = 0.00355
           self.R = 0.01065
           self.popt_in = [6.29230105e+03, 7.24254548e+07]
           self.popt_out = [2.13704667e+03, 1.00960705e+08]
        elif self.r_max == 0.018:
           self.r = 0.004
           self.R = 0.012
           self.popt_in = [5.66912918e+03, 7.66378929e+07]
           self.popt_out = [3.04617628e+03, 9.46300951e+07]
        elif self.r_max == 0.02:
           self.r = 0.00445 
           self.R = 0.01335
           self.popt_in = [5.36141778e+03, 7.87273868e+07]
           self.popt_out = [3.44279531e+03, 9.18833710e+07]

    def value_shape(self):
        return (3, )


class SymetricSinus(df.UserExpression):
    def __init__(self, r_max, shift=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if shift is None:
            self.shift = 0
        else:
            self.shift = shift
        self.R = r_max
        self.Win = 0.012
        self.Wout = 0.013
        self.L0 = 0.022
        self.L1 = 0.012
        self.dx = 0.0005

        mx = [-self.L0, -self.L1, 0,  self.L1, self.L0]
        my = [self.Win, self.Win, self.R, self.Wout,  self.Wout]

        cs = CubicSpline(mx[1:-1], my[1:-1])
        self.cs = cs

        p0 = self.L1 + self.dx
        p1 = self.L1 - self.dx
        self.p0 = p0
        self.p1 = p1

        self.hs0 = CubicHermiteSpline(
            [-p0, -p1], [self.Win, cs(-p1)], [0, cs.derivative()(-p1)]
        )
        self.hs1 = CubicHermiteSpline(
            [p1, p0], [cs(p1), self.Wout], [cs.derivative()(p1), 0]
        )

    def shape(self, x):
        if x < -self.p0:
            y = self.Win
        elif x < -self.p1:
            y = self.hs0(x)
        elif x < self.p1:
            y = self.cs(x)
        elif x < self.p0:
            y = self.hs1(x)
        else:
            y = self.Wout
        return y

    def derivative(self, x):
        if x < -self.p0:
            y = 0.0
        elif x < -self.p1:
            y = self.hs0.derivative()(x)
        elif x < self.p1:
            y = self.cs.derivative()(x)
        elif x < self.p0:
            y = self.hs1.derivative()(x)
        else:
            y = 0.0
        return y

    def derivative_as_expression(self):
        return Eval(self.derivative, degree=2)


    def eval(self, value, x):

        r = df.sqrt(x[0]**2 + x[1]**2)
        r_displacement = self.shape(x[2]) + self.shift

        value[0] = (
            x[0] * r_displacement / r - x[0]
        )

        value[1] = (
            x[1] * r_displacement / r - x[1]
        )

        value[2] = 0

    def value_shape(self):
        return (3, )



def u_init_valv(mesh: df.Mesh, maximal_radius: float, radius:float, radius2: float, bndry_marker, labels_bndry, order: int = 2):
    r_max = maximal_radius
    # r_max = 0.016
    # r_max = 0.018
    # r_max = 0.02
    u_init_interface = UInit(r_max, scale=1)
    u_init_out = UInit(r_max, scale=1 + radius2 / radius)
    space = df.VectorFunctionSpace(mesh, "CG", order)
    u = df.TrialFunction(space)
    u_ = df.TestFunction(space)
    function = df.Function(space)

    form = df.inner(df.grad(u), df.grad(u_)) * df.dx + 10 * df.div(u)*df.div(u_) * df.dx
    rhs = df.inner(df.Constant((0.0, 0.0, 0.0)), u_) * df.dx
    zero = df.Constant((0.0, 0.0, 0.0))
    bcs = [
        df.DirichletBC(space.sub(2), 0, bndry_marker, labels_bndry["inflow_f"]),
        df.DirichletBC(space.sub(2), 0, bndry_marker, labels_bndry["outflow_f"]),
        df.DirichletBC(space.sub(2), 0, bndry_marker, labels_bndry["inflow_s"]),
        df.DirichletBC(space.sub(2), 0, bndry_marker, labels_bndry["outflow_s"]),
        df.DirichletBC(space, u_init_interface, bndry_marker, labels_bndry["mid_f"]),
        df.DirichletBC(space, u_init_interface, bndry_marker, labels_bndry["mid_s"]),
        df.DirichletBC(space, u_init_out, bndry_marker, labels_bndry["out"]),
    ]
    problem = df.LinearVariationalProblem(form, rhs, function, bcs)
    solver = df.LinearVariationalSolver(problem)
    solver.solve()
    return function


def u_init_symmetric(mesh: df.Mesh, r_max: float, radius:float, radius2: float, bndry_marker, labels_bndry, order: int = 2):
    u_init_interface = SymetricSinus(r_max, shift=0)
    u_init_out = SymetricSinus(r_max, shift=radius2)
    space = df.VectorFunctionSpace(mesh, "CG", order)
    u = df.TrialFunction(space)
    u_ = df.TestFunction(space)
    function = df.Function(space)

    form = df.inner(df.grad(u), df.grad(u_)) * df.dx + 10 * df.div(u)*df.div(u_) * df.dx
    rhs = df.inner(df.Constant((0.0, 0.0, 0.0)), u_) * df.dx
    zero = df.Constant((0.0, 0.0, 0.0))
    bcs = [
        df.DirichletBC(space.sub(2), 0, bndry_marker, labels_bndry["inflow_f"]),
        df.DirichletBC(space.sub(2), 0, bndry_marker, labels_bndry["outflow_f"]),
        df.DirichletBC(space.sub(2), 0, bndry_marker, labels_bndry["inflow_s"]),
        df.DirichletBC(space.sub(2), 0, bndry_marker, labels_bndry["outflow_s"]),
        df.DirichletBC(space, u_init_interface, bndry_marker, labels_bndry["mid_f"]),
        df.DirichletBC(space, u_init_interface, bndry_marker, labels_bndry["mid_s"]),
        df.DirichletBC(space, u_init_out, bndry_marker, labels_bndry["out"]),
    ]
    problem = df.LinearVariationalProblem(form, rhs, function, bcs)
    solver = df.LinearVariationalSolver(problem)
    solver.solve()
    return function
