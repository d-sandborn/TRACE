# Column inventories

A column integration function is available:

```python
from tracepy import integrate_column
```

This function integrates concentrations (e.g. anthropogenic carbon) for a single location between user-provided depths via Piecewise Cubic Hermite Interpolating Polynomial ([PCHIP](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html)) followed by [Romberg](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.romb.html) numerical integration. While this function only calculates a column inventory at a single location, it is designed to be looped to produce regional or global inventories. 

```python
>>> integrate_column?
```
```
Signature:
integrate_column(
    integrand,
    salinity,
    temperature,
    depth,
    lat: float,
    bottom: float,
    top: float = 0,
    romb_resolution: int = 10,
)
Docstring:
Integrates anthropogenic carbon estimates at given geographic location.

This function may be looped to construct basin or global inventories.

Parameters
----------
integrand : numpy.ndarray
    Array with length n of quantities to integrate.
    Must be in per kg seawater units.
salinity : numpy.ndarray
    Array with length n of salinity values associated with integrand.
    Must be practical scale.
temperature : numpy.ndarray
    Array with length n of temperature values associated with integrand.
    Must be in-situ temperature.
depth : numpy.ndarray
    Array with length n of depth values associated with integrand.
    Must be units of meters bsl (i.e. positive values).
lat : float
    Latitude N of column inventory location in degrees. Used for depth
    to pressure conversion.
bottom : float
    Maximum depth of integration in meters bsl.
top : float, optional
    Minimum depth of integration in meters bsl.
    The default is 0.
romb_resolution : int, optional
    Controls of points over which pchip interpolation interpolates
    integrand, using formula points = 2^romb_resolution-1.
    The default is 10.

Returns
-------
column_inventory : float
    Column integration of integrand, given in input units transformed to
    per square meter.
```