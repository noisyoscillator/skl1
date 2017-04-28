from setuptools import setup, Extension
from Cython.Build import cythonize

skl1_ext = cythonize(Extension("skl1._euler",
                     sources=["skl1/threefry.c", "skl1/_euler.pyx"],
                     include_dirs=["skl1"],
                     ))

setup(name='skl1',
      version='0.1.0.dev0',
      description='Numerical simulation of the Langevin equation',
      author='Pierre de Buyl',
      license='BSD',
      packages=['skl1'],
      ext_modules = skl1_ext,
      setup_requires=['cython'],
      )
