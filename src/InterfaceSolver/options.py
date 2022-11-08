from petsc4py import PETSc


def opts_setup(opts_dict):
    """
    Set PETSc options from dic `opts_dict`.

    *Arguments*

        opts_dics `dics`
    """
    opts = PETSc.Options()
    for key, value in opts_dict.items():
        if isinstance(value, dict):
            opts.prefixPush(key)
            opts_setup(value)
            opts.prefixPop()
        else: opts[key]=value

DEFAULT_OPTIONS = {
    'snes_': {
        'rtol': 1.e-10,
        'atol': 1.e-10,
        'stol': 1.e-10,
        'max_it': 10
    },
    'pc_': {
        'type': 'lu',
        'factor_mat_solver_type': 'mumps'
    },
    'mat_': {
        'mumps_': {
            'cntl_1': 1e-8,
            'icntl_14': 100,
            'icntl_24':1
        }
    },
    'ksp_': {
        'type': 'preonly'
    },
}