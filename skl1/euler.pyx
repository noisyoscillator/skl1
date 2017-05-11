import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from libc.stdint cimport uint64_t

from threefry cimport rng

cdef class moments_t:
    cdef double mu
    cdef double v
    cdef int n
    def __cinit__(self):
        self.mu = 0
        self.v = 0
        self.n = 0

    cdef append(self, x):
        cdef double mu = self.mu
        self.n = self.n + 1
        self.mu = mu + (x-mu)/self.n
        self.v = self.v + (x-self.mu)*(x-mu)

    def mu_v(self):
        return self.mu, self.v/self.n

cdef class cyfunc_d_d:
    cpdef double force(self, double x):
        return 0
    def __cinit__(self):
        pass


cdef class pyfunc_d_d(cyfunc_d_d):
    cdef object py_force
    cpdef double force(self, double x):
        return self.py_force(x)
    def __init__(self, force):
        self.py_force = force


cdef class linear_friction(cyfunc_d_d):
    cdef double gamma
    def __init__(self, gamma):
        self.gamma = gamma
    cpdef double force(self, double x):
        return -self.gamma*x


cdef integrate_inner(double *x, double *v, double D, double dt, int interval, cyfunc_d_d f, cyfunc_d_d g, rng gen, moments_t m):
    cdef int i
    cdef double in_x  = x[0]
    cdef double in_v = v[0]
    cdef double x_tmp, noise

    noise = sqrt(2*D*dt)
    for i in range(interval):
        x_tmp = in_x
        in_x = in_x + in_v*dt
        in_v = in_v + (f.force(x_tmp) + g.force(in_v))*dt + noise*gen.random_normal()
        m.append(in_v)
    x[0] = in_x
    v[0] = in_v


def integrate(double x, double v, double D, double dt, int interval, int steps, f=None, g=None, seed=None, callback=None):
    cdef cyfunc_d_d py_f, py_g
    cdef int i
    cdef moments_t moments
    r = rng(seed)
    moments = moments_t()

    if f is None:
        py_f = cyfunc_d_d()
    elif isinstance(f, cyfunc_d_d):
        py_f = f
    elif callable(f):
        py_f = pyfunc_d_d(f)
    else:
        raise ValueError("f should be a callable or a cyfunc_d_d")

    if g is None:
        py_g = cyfunc_d_d()
    elif isinstance(g, int) or isinstance(g, float):
        py_g = linear_friction(g)
    elif isinstance(g, cyfunc_d_d):
        py_g = g
    elif callable(g):
        py_g = pyfunc_d_d(g)
    else:
        raise ValueError("g should be a callable or a cyfunc_d_d")

    cdef double[:] x_out = np.empty(steps, dtype=float)
    cdef double[:] v_out = np.empty(steps, dtype=float)

    for i in range(steps):
        integrate_inner(&x, &v, D, dt, interval, py_f, py_g, r, moments)
        x_out[i] = x
        v_out[i] = v
        if callback is not None:
            x, v = callback(x, v)

    return np.asarray(x_out), np.asarray(v_out), moments
