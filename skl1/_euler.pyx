import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from libc.stdint cimport uint64_t

cdef extern from "threefry.h":
    ctypedef struct threefry_t:
        uint64_t c0
        uint64_t c1

    double threefry_double(threefry_t *c, threefry_t *k)


cdef class rng:
    cdef threefry_t counter
    cdef threefry_t key

    def __cinit__(self):
        pass

    cpdef set(self, counter, key):
        self.counter.c0 = counter
        self.counter.c1 = 0
        self.key.c0 = key[0]
        self.key.c1 = key[1]

    def __init__(self, counter, key):
        self.set(counter, key)

    cdef double random_normal(self):
        cdef int i
        cdef double x
        x = 0
        for i in range(12):
            x = x + (threefry_double(&self.counter, &self.key)-0.5)
        return x


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


def integrate(double x, double v, double D, double dt, int steps, int n, f, g, counter, key, callback=None):
    cdef MF_py py_f, py_g
    cdef int i
    r = rng(counter, key)

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
