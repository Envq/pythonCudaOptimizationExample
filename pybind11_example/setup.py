from pathlib import Path
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


cuda_acceleration_module = Pybind11Extension(
    "cuda_accelerations",
    [str(fname) for fname in Path('src').glob('*.cpp')],
    include_dirs=['include'],
    # extra_compile_args=['-O3']
)

setup(
    name="cuda_accelerations",
    version=0.1,
    ext_modules=[cuda_acceleration_module],
    cmdclass={"build_ext": build_ext},
)