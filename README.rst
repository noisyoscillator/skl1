skl1
====

Tentative "scikit Langevin" module.

Aims
----

1. Provide a modular solver for Langevin equations of the type ``dx/dt = v``, ``dv/dt
   = -gamma v + force + noise`` or similar overdamped systems.
2. Accept user-defined functions for the forces, either in Python or in Cython.


Installation
------------

Dependencies: Python, `NumPy <http://www.numpy.org/>`_, `Cython <http://cython.org/>`_,
`threefry <https://github.com/pdebuyl/threefry>`_.


    pip3 install git+https://github.com/pdebuyl/skl1

or

    python3 -m pip install git+https://github.com/pdebuyl/skl1

