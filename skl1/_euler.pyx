import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from libc.stdint cimport uint64_t

from threefry cimport rng

cdef class MyFunctionClass:
    cpdef double force(self, double x):
        return 0


cdef integrate_inner(double *x, double *v, double D, double dt, int n, MyFunctionClass f, MyFunctionClass g, rng gen):
    cdef int i
    cdef double in_x  = x[0]
    cdef double in_v = v[0]
    cdef double x_tmp, noise

    noise = sqrt(2*D*dt)
    for i in range(n):
        x_tmp = in_x
        in_x = in_x + in_v*dt
        in_v = in_v + (f.force(x_tmp) + g.force(in_v))*dt + noise*gen.random_normal()
    x[0] = in_x
    v[0] = in_v


cdef class MF_py(MyFunctionClass):
    cdef object py_force
    cpdef double force(self, double x):
        return self.py_force(x)
    def __init__(self, force):
        self.py_force = force


def integrate(double x, double v, double D, double dt, int steps, int n, f, g, seed):
    cdef MF_py py_f, py_g
    cdef int i
    r = rng(seed)

    if isinstance(f, MyFunctionClass):
        py_f = f
    elif callable(f):
        py_f = MF_py(f)
    else:
        raise ValueError("f should be a callable or a MyFunctionClass")

    if isinstance(g, MyFunctionClass):
        py_g = g
    elif callable(g):
        py_g = MF_py(g)
    else:
        raise ValueError("g should be a callable or a MyFunctionClass")

    cdef double[:] x_out = np.empty(n, dtype=float)
    cdef double[:] v_out = np.empty(n, dtype=float)

    for i in range(n):
        integrate_inner(&x, &v, D, dt, steps, py_f, py_g, r)
        x_out[i] = x
        v_out[i] = v

    return np.asarray(x_out), np.asarray(v_out)
