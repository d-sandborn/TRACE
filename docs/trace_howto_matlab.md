# Using TRACEv1

This function needs the CSIRO seawater package to run. Scale differences from TEOS-10 are a negligible component of estimate errors as formulated. For demos see the **MATLAB Examples** page. 

!!! inputs "`TRACEv1` Arguments"

	Array dimensions:

	- p: Integer number of desired estimate types. For TRACE this is always 1. 
	- n: Integer number of desired estimate locations 
	- e: Integer number of equations used at each location (up to 4) 
	- y: Integer number of parameter measurement types provided by the user. 
	- n*e: Total number of estimates returned as an n by e array

    ### Coordinates

    Each of n rows of output coordinates and dates indicates a single location in space/time at which an estimation is made. `trace` cannot presently handle multidimensional (e.g. latitude/longitude/depth) arrays of coordinates: they must be [flattened](https://numpy.org/doc/stable/reference/generated/numpy.ravel.html) to one-dimensional vectors, then concatenated into the columns as described below. 

    * `OutputCoordinates` (required n by 3 array): Coordinates at which estimates are desired. The first column should be longitude (degrees E), the second column should be latitude (degrees N), and the third column should be depth (m).

    * `Dates` (required n by 1 array): Year c.e. Decimals are allowed, but the information contained in the decimal is disregarded because the technique uncertainties are too great to resolve the impacts of fractional years.

    ### Predictors

    Unlike for ESPER, only salinity and temperature are accepted predictors for age and other preformed properties. 

    * `PredictorMeasurements` (required n by y array): Parameter measurements that will be used to estimate alkalinity. The column order (y columns) is arbitrary, but specified by PredictorTypes. Temperature should be expressed as degrees C and salinity should be specified with the unitless convention. NaN inputs are acceptable, but will lead to NaN estimates for any equations that depend on that parameter. If temperature is not provided then it will be estimated from salinity (not recommended).

    * `PredictorTypes` (required 1 by y vector): Vector indicating which parameter is placed in each column of the 'PredictorMeasurements' input. Note that salinity is required for all equations. Input parameter key:
        * `1`. Salinity
        * `2`. Temperature

    ### Atmospheric CO2 

    * `AtmCO2Trajectory` (integer between 1 and 9): This specifies the atmospheric CO2Trajectory. There are several options between Historical and SSPs. Historical are derived from [this website](https://www.ncei.noaa.gov/access/paleo-search/study/10437) supplemented from [here](https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_mlo.txt) after 1959. Historical is currently updated through 2022 and assumes a linear increase in the years that follow (recommended that you update the underlying data file as needed). SSPs are from the supplement to [Meinshausen et al. 2020](https://gmd.copernicus.org/articles/13/3571/2020/). RCPs are not currently supported, but can be added to the file if the user desires
        * `1`. Historical/Linear **(default)**
        * `2`. SSP1_1.9
        * `3`. SSP1_2.6
        * `4`. SSP2_4.5
        * `5`. SSP3_7.0
        * `6`. SSP3_7.0_lowNTCF
        * `7`. SSP4_3.4
        * `8`. SSP4_6.0
        * `9`. SSP5_3.4_over
    
        Custom columns can be added to the data/CO2TrajectoreisAdjusted.txt file and referenced here.

    ### Optional inputs

    All remaining inputs must be specified as sequential input argument pairs (e.g. "..., 'Equations',[1:4], etc...")
    
    * `PreindustrialxCO2` (optional 1 by 1 scalar, default 280 ppm): Allows the user to specify an arbitrary reference xCO2 for the preindustrial era in units of ppm. Higher numbers will result in lower Canth estimates including negative values. Small negative estimates are also possible with the default input.

    * `VerboseTF` (Optional boolean, default true): Setting this to false will make TRACE stop printing updates to the command line. This behavior can also be permanently disabled below. Warnings and errors, if any, will be given regardless.

    * `MeasUncerts` (Optional n by y array or 1 by y vector, default: [0.003 S, 0.003 degrees C (T or theta), 1AOU]: Array of measurement uncertainties (see 'PredictorMeasurements' for units). Uncertainties should be presented in order indicated by PredictorTypes. Providing these estimates will improve estimate uncertainties in 'UncertaintyEstimates'. Measurement uncertainties are a small part of TRACE estimate uncertainties for WOCE-quality measurements. However, estimate uncertainty scales with measurement uncertainty, so it is recommended that measurement uncertainties be specified for sensor measurements. If this optional input argument is not provided, the default WOCE-quality uncertainty is assumed. If a 1 by y array is provided then the uncertainty estimates are assumed to apply uniformly to all input parameter measurements.

!!! outputs "`TRACEv1` Output"

	`OutputEstimates`: A n by e array of TRACE estimates specific to the coordinates and parameter measurements provided as inputs. Units are micromoles per kg.
	
	`UncertaintyEstimates`: A n by e array of uncertainty estimates specific to the coordinates, parameter measurements, and parameter uncertainties provided.
	
	`Missing data`: should be indicated with a NaN. A NaN coordinate will yield NaN estimates for all equations at that coordinate. A NaN parameter value will yield NaN estimates for all equations that require that parameter.
