#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="trace-python",
    python_requires=">=3.12",
    version="1.1.3",
    description="Tracer-based Rapid Anthropogenic Carbon Estimation (TRACE) in Python",
    long_description="After [TRACEv1](https://github.com/BRCScienceProducts/TRACEv1) and being developed in parallel with [ESPER](https://github.com/BRCScienceProducts/ESPER) and [PyESPER](https://github.com/LarissaMDias/PyESPER). Please reference the TRACEv1 [publication](https://doi.org/10.5194/essd-17-3073-2025) for further details.",
    author="Daniel Sandborn, Brendan Carter, Reese Barrett, Colette Kelly",
    author_email="sandborn@uw.edu",
    url="https://github.com/d-sandborn/TRACE",
    include_package_data=True,  # make sure the data files are installed
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "netcdf4",
        "xarray",
        "scipy",
        "setuptools",
        "matplotlib",
        "pyyaml",
        "tqdm",
        "seawater",  # deprecated, kept for comparability with TRACEv1
        "gsw",  # replacement for seawater
        "PyCO2SYS @ git+https://github.com/mvdh7/PyCO2SYS@v2.0.0b9",  # to be replaced with production version
        "numba",
        "shapely",
        "geopandas",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.13",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Oceanography",
    ],
)
