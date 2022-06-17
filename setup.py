'''Installs cpp_patcher'''
from os import path
from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension


ext_modules = [
    Pybind11Extension(
        "npy_patcher",
        sorted(glob("src/*.cpp")),
        include_dirs=['./'],
        cxx_std=17
    ),
]


version = '1.0.8'
this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='npy-patcher',
    version=version,
    description='C++ based NumPy N-Dimesional patch extraction.',
    author='Matthew Lyon',
    author_email='matthewlyon18@gmail.com',
    url='https://github.com/m-lyon/npy-cpp-patches',
    download_url=f'https://https://github.com/m-lyon/npy-cpp-patches/archive/v{version}.tar.gz',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.6',
    license='MIT License',
    classifiers=[
        'Programming Language :: Python',
        'Operating System :: Unix',
        'Operating System :: MacOS',
    ],
    ext_modules=ext_modules
)
