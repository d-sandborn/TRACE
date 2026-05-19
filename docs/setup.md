# Installing TRACE

TRACE is presently available for MATLAB (as TRACEv1) and Python (as TRACE-Python). Both interfaces produce identical estimates and inventories within machine precision, and operate with comprable speed. 

=== "Python"

    Download and unzip a release from the [TRACE-Python repository](https://github.com/d-sandborn/pyTRACE/releases).  Ensure pip and python are installed in a virtual environment (we suggest [this method](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html), but any of the mainline methods should work). TRACE and all of its dependencies can then be installed by navigating to the unzipped directory of TRACE and running the following in a terminal emulator:

	```bash
	python -m pip install -e .
	```
	
	!!! note "pip and conda"
	
		TRACE will be made available via PyPI once PyCO2SYS >=2 is available there. PyCO2SYS >= v2 is required for speed and stability, and a beta version is installed by default using the command above. More information on that package can be found [here](https://mvdh.xyz/PyCO2SYS/). TRACE is not yet available via conda-forge, but this is a target for future development if interest warrants it. 

	Finished installing? [Learn how to use TRACE-Python here.](https://d-sandborn.github.io/TRACE/trace_howto/)

=== "MATLAB"

	Download and unzip a release from the [TRACEv1 repository](https://github.com/BRCScienceProducts/TRACEv1) to your computer and extract to a location on your MATLAB path (or [add the location to your path](https://www.mathworks.com/help/matlab/ref/addpath.html)). Ensure the `TRACEv1.m` function and the `private` directory are in the same folder. Several toolboxes may need to be installed if they are not already. 
	
	Finished installing? [Learn how to use TRACEv1 here.](https://d-sandborn.github.io/TRACE/trace_howto_matlab/)