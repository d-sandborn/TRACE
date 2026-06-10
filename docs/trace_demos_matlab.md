# MATLAB Demos

This first example asks for estimates at the surface ocean at the equator/prime meridian in the years 2000 and 2200 assuming SSP5_3.4_over is followed

```MATLAB
[Canth]=TRACEv1([0 0 0;0 0 0],[2000;2200],[35 20;35 20],[1 2],[9],[0])
```

Results in

```MATLAB
Canth = 47.7869 79.8749
```

This second example demonstrates a function call performed without providing temperature information, which is not recommended and should result in a warning

```MATLAB
[Canth]=TRACEv1([0 0 0;0 0 0],[2000;2010],[35;35],[1],[1],[0])
```

Results in

```MATLAB
Warning: TRACE was called either without providing temperature or without specifying which column of PredictorMeasurements contains temperature. Temperature is therefore being estimated from salinity and coordinate information, but this is not optimal and the validation for TRACE should not be considered appropriate for the estimates returned from this function call.

    In TRACEv1 (line 325)

Canth = 56.0591 66.4567
```
