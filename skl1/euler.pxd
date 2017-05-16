
from threefry cimport rng
cimport skl1.core
#from skl1.core cimport moments_t, cyfunc_d_d, cyfund_nd

cdef integrate_inner(double *x, double *v, double D, double dt, int interval, skl1.core.cyfunc_d_d f, skl1.core.cyfunc_d_d g, rng gen, skl1.core.moments_t m)
