# Column inventories

A column integration function is available:

```python

from tracepy import column_integration

```

This function integrates concentrations (e.g. anthropogenic carbon) for a single location between user-provided depths via Piecewise Cubic Hermite Interpolating Polynomial ([PCHIP](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html)) followed by [Romberg](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.romb.html) numerical integration. While this function only calculates a column inventory at a single location, it is designed to be looped to produce regional or global inventories. 