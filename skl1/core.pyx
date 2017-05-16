
cdef class moments_t:
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
    cpdef double force(self, double x):
        return self.py_force(x)
    def __init__(self, force):
        self.py_force = force


cdef class linear_friction(cyfunc_d_d):
    def __init__(self, gamma):
        self.gamma = gamma
    cpdef double force(self, double x):
        return -self.gamma*x

cdef class cyfunc_nd:
    cpdef void force(self, double[::1] x, double[::1] f):
        cdef int i
        for i in range(f.shape[0]):
            f[i] = 0
    def __init__(self):
        pass

cdef class pyfunc_nd(cyfunc_nd):
    cpdef void force(self, double[::1] x, double[::1] f):
        self.py_force(x, f)
    def __init__(self, force):
        self.py_force = force


cdef class linear_friction_nd(cyfunc_nd):
    def __init__(self, gamma):
        self.gamma = gamma
    cpdef void force(self, double[::1] x, double[::1] f):
        cdef int i
        for i in range(x.shape[0]):
            f[i] = -self.gamma*x[i]

