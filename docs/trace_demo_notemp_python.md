# Estimation without temperature

Calling ```trace``` without a temperature input will cause it to estimate temperature from salinity and coordinates via a neural network. This is **not** recommended, but will yield results (with a warning) as in TRACEv1:

```python
output = trace(
    output_coordinates=np.array([[0, 0, 0], [0, 0, 0]]),
    dates=np.array([2000, 2010]),
    predictor_measurements=np.array([[35], [35]]),
    predictor_types=np.array([1]),
    atm_co2_trajectory=1
)

UserWarning: Temperature is being estimated from salinity and coordinate information.

>>> output

<xarray.Dataset> Size: 448B
Dimensions:           (loc: 2)
Coordinates:
    year              (loc) <U20 160B '2000-01-01T00:00:00Z' '2010-01-01T00:0...
    lon               (loc) float64 16B 0.0 0.0
    lat               (loc) float64 16B 0.0 0.0
    depth             (loc) float64 16B 0.0 0.0
Dimensions without coordinates: loc
Data variables: (12/15)
    canth             (loc) float64 16B 55.32 65.56
    mean_age          (loc) float64 16B 1.883 1.883
    mode_age          (loc) float64 16B 0.4424 0.4424
    dic               (loc) float64 16B 1.941e+03 1.951e+03
    dic_ref           (loc) float64 16B 1.885e+03 1.885e+03
    pco2              (loc) float64 16B 346.2 362.9
               ...
    preformed_p       (loc) float64 16B 0.08551 0.08551
    temperature       (loc) float64 16B 26.47 26.47
    salinity          (loc) float64 16B 35.0 35.0
    u_canth           (loc) float64 16B 9.603 10.96
    delta_over_gamma  (loc) float64 16B 1.304 1.304
    scale_factors     (loc) float64 16B 0.01341 0.01341
Attributes:
    Conventions:        CF-1.10
    description:        Results of Tracer-based Rapid Anthropogenic Carbon Es...
    history:            TRACE for Python, 2026-10-22 16:01:09.623526 Python 3...
    date_created:       2026-10-22 16:01:09.623549
    references:         doi.org/10.5194/gmd-19-5961-2026
    co2sys_parameters:  CO2System settings.\n├─ EQUILIBRIUM CONSTANTS:\n│  ├─...
    trace_parameters:   per_kg_sw_tf: True, canth_diseq: 1.0, eos: seawater, ...

>>> output.canth.data

array([55.32020202, 65.55609807])

```

The same result was obtained in TRACEv1 for the same inputs.