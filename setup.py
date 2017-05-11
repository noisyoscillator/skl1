from setuptools import setup, Extension
from Cython.Build import cythonize

try:
    import threefry
    threefry_include = threefry.get_include()
except:
    threefry_include = ''

skl1_ext = cythonize(Extension("skl1.euler",
                     sources=["skl1/euler.pyx"],
                     include_dirs=["skl1", threefry_include],
                     ))

setup(name='skl1',
      version='0.1.0.dev0',
      description='Numerical simulation of the Langevin equation',
      author='Pierre de Buyl',
      license='BSD',
      packages=['skl1'],
      ext_modules = skl1_ext,
      setup_requires=['cython', 'threefry'],
      )
