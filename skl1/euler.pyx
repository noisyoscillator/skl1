import numpy as np
cimport numpy as np
from libc.math cimport sqrt, cos, sin
from libc.stdint cimport uint64_t

from threefry cimport rng

from skl1.core cimport moments_t, cyfunc_d_d, pyfunc_d_d, linear_friction, cyfunc_nd, pyfunc_nd, linear_friction_nd

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
    cdef double time

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
        time = i*interval*dt
        x_out[i] = x
        v_out[i] = v
        if callback is not None:
            x, v = callback(x, v, time)

    return np.asarray(x_out), np.asarray(v_out), moments




def integrate_nd(double[::1] x, double[::1] v, double D, double dt, int interval, int steps, f=None, g=None, seed=None):
    cdef cyfunc_nd cy_f, cy_g
    cdef int t_idx1, t_idx2, j, n_dims

    assert x.shape[0]==v.shape[0]

    r = rng(seed)

    if f is None:
        cy_f = cyfunc_nd()
    elif isinstance(f, cyfunc_nd):
        cy_f = f
    elif callable(f):
        cy_f = pyfunc_nd(f)
    else:
        raise ValueError("f should be a callable or a cyfunc_nd")

    if g is None:
        cy_g = cyfunc_nd()
    elif isinstance(g, int) or isinstance(g, float):
        cy_g = linear_friction_nd(g)
    elif isinstance(g, cyfunc_nd):
        cy_g = g
    elif callable(g):
        cy_g = pyfunc_nd(g)
    else:
        raise ValueError("g should be a callable or a cyfunc_nd")

    noise = sqrt(2*D*dt)
    n_dims = x.shape[0]

    cdef double[::1] force = np.zeros(n_dims)

    cdef double[:,::1] x_out = np.empty((steps, n_dims), dtype=float)
    cdef double[:,::1] v_out = np.empty((steps, n_dims), dtype=float)

    for t_idx1 in range(steps):
        for t_idx2 in range(interval):
            for j in range(n_dims):
                x[j] = x[j] + v[j]*dt
            cy_f.force(x, force)
            for j in range(n_dims):
                v[j] = v[j] + force[j]*dt
            cy_g.force(v, force)
            for j in range(n_dims):
                v[j] = v[j] + force[j]*dt + noise*r.random_normal()

        x_out[t_idx1] = x
        v_out[t_idx1] = v

    return np.asarray(x_out), np.asarray(v_out)

def integrate_OD_2d_theta(double[::1] x, double th, double mu, double D, double v0, double Dr, double dt, int interval, int steps, f=None, seed=None):
    cdef cyfunc_nd cy_f, cy_g
    cdef int t_idx1, t_idx2, j, n_dims
    cdef double noise, th_noise

    assert x.shape[0]==2

    r = rng(seed)

    if f is None:
        cy_f = cyfunc_nd()
    elif isinstance(f, cyfunc_nd):
        cy_f = f
    elif callable(f):
        cy_f = pyfunc_nd(f)
    else:
        raise ValueError("f should be a callable or a cyfunc_nd")

    noise = sqrt(2*D*dt)
    th_noise = sqrt(2*Dr*dt)
    n_dims = x.shape[0]

    cdef double[::1] force = np.zeros(n_dims)

    cdef double[:,::1] x_out = np.empty((steps, n_dims), dtype=float)
    cdef double[::1] th_out = np.empty((steps,), dtype=float)

    for t_idx1 in range(steps):
        for t_idx2 in range(interval):
            cy_f.force(x, force)
            th = th + th_noise*r.random_normal()
            x[0] = x[0] + (mu*force[0] + cos(th)*v0)*dt + noise*r.random_normal()
            x[1] = x[1] + (mu*force[1] + sin(th)*v0)*dt + noise*r.random_normal()

        x_out[t_idx1] = x
        th_out[t_idx1] = th

    return np.asarray(x_out), np.asarray(th_out)
