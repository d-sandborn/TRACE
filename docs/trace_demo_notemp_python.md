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

output

<xarray.Dataset> Size: 448B
Dimensions:           (loc: 2)
Coordinates:
    year              (loc) <U20 160B '2000-01-01T00:00:00Z' '2010-01-01T00:0...
    lon               (loc) int64 16B 0 0
    lat               (loc) int64 16B 0 0
    depth             (loc) int64 16B 0 0
Dimensions without coordinates: loc
Data variables:
    canth             (loc) float64 16B 56.06 66.46
    mean_age          (loc) float64 16B 1.883 1.883
    mode_age          (loc) float64 16B 0.4424 0.4424
    dic               (loc) float64 16B 1.925e+03 1.935e+03
    dic_ref           (loc) float64 16B 1.869e+03 1.869e+03
    pco2              (loc) float64 16B 322.4 337.9
    pco2_ref          (loc) float64 16B 252.0 252.0
    preformed_ta      (loc) float64 16B 2.282e+03 2.282e+03
    preformed_si      (loc) float64 16B 1.787 1.787
    preformed_p       (loc) float64 16B 0.08551 0.08551
    temperature       (loc) float64 16B 26.47 26.47
    salinity          (loc) float64 16B 35.0 35.0
    u_canth           (loc) float64 16B 9.699 11.08
    delta_over_gamma  (loc) float64 16B 1.3 1.3
    scale_factors     (loc) float64 16B 0.01341 0.01341
Attributes:
    Conventions:        CF-1.10
    description:        Results of Tracer-based Rapid Anthropogenic Carbon Es...
    history:            TRACE version 0.2.0 (beta), 2025-07-16 19:36:55.79208...
    date_created:       2025-07-16 19:36:55.792097
    references:         doi.org/10.5194/essd-2024-560
    co2sys_parameters:  opt_pH_scale: 1, opt_k_carbonic: 10, opt_k_HSO4: 1, o...
    trace_parameters:   per_kg_sw_tf: True, canth_diseq: 1.0, eos: seawater, ...

output.canth.data

array([56.059132, 66.45668126])

```

The same result was obtained in TRACEv1: C<sub>anth</sub> = ```[56.0591 66.4567]``` for the same inputs.