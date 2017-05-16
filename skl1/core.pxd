
cdef class moments_t:
    cdef double mu
    cdef double v
    cdef int n
    cdef append(self, x)


cdef class cyfunc_d_d:
    cpdef double force(self, double x)


cdef class pyfunc_d_d(cyfunc_d_d):
    cdef object py_force
    cpdef double force(self, double x)


cdef class linear_friction(cyfunc_d_d):
    cdef double gamma
    cpdef double force(self, double x)


cdef class cyfunc_nd:
    cpdef void force(self, double[::1] x, double[::1] f)


cdef class pyfunc_nd(cyfunc_nd):
    cdef object py_force
    cpdef void force(self, double[::1] x, double[::1] f)

cdef class linear_friction_nd(cyfunc_nd):
    cdef double gamma
