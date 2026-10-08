# Other transient tracers

The age models produced by `trace` can be combined with the age history of an arbitrary transient traces to yield estimates of the atmospheric transient invasion of the ocean. So far, only nitrous oxide (N<sub>2</sub>O) has been implemented. The routine first runs `trace` and then applies the TTD to a user-specified N<sub>2</sub>O trajectory. 

!!! danger "Experimental"

    This routine is experimental, and is subject to further validation and peer-review in a manuscript in preparation.

The relevant output quantities are:

- `n2o`: Preformed nitrous oxide equilibrium concentration, in nmol kg<sup>-1</sup>.
- `n2o_ref`: Preindustrial preformed nitrous oxide equilibrium concentration, in nmol kg<sup>-1</sup>.
- `pn2o`: Preformed nitrous oxide equilibrium partial pressure, in natm.
- `pn2o_ref`: Preindustrial preformed nitrous oxide equilibrium partial pressure, in natm.
- `n2o_prime`: Concentration of atmospheric transient nitrous oxide, equal to (**n2o**-**n2o_ref**). This is analagous to C<sub>anth</sub>, except that the imperfectly-transient nature of nitrous oxide precludes denoting this fraction "anthropogenic".

```python
output = trace_n2o(
    output_coordinates=np.array([[0, 0, 0], [0, 0, 0]]),
    dates=np.array([2000, 2010]),
    predictor_measurements=np.array([[35], [35]]),
    predictor_types=np.array([1]),
    atm_co2_trajectory=1
)

>>> output

<xarray.Dataset> Size: 528B
Dimensions:           (loc: 2)
Coordinates:
    year              (loc) <U20 160B '2000-01-01T00:00:00Z' '2010-01-01T00:0...
    lon               (loc) float64 16B 0.0 0.0
    lat               (loc) float64 16B 0.0 0.0
    depth             (loc) float64 16B 0.0 0.0
Dimensions without coordinates: loc
Data variables: (12/20)
    canth             (loc) float64 16B 56.06 66.46
    mean_age          (loc) float64 16B 1.883 1.883
    mode_age          (loc) float64 16B 0.4424 0.4424
    dic               (loc) float64 16B 1.925e+03 1.935e+03
    dic_ref           (loc) float64 16B 1.869e+03 1.869e+03
    pco2              (loc) float64 16B 322.4 337.9
               ...
    scale_factors     (loc) float64 16B 0.01341 0.01341
    n2o               (loc) float64 16B 5.508 5.641
    n2o_ref           (loc) float64 16B 4.753 4.753
    pn2o              (loc) float64 16B 282.6 289.4
    pn2o_ref          (loc) float64 16B 243.9 243.9
    n2o_prime         (loc) float64 16B 0.755 0.8876
Attributes:
    Conventions:        CF-1.10
    description:        Results of Tracer-based Rapid Anthropogenic Carbon Es...
    history:            TRACE version 1.1.0, 2026-09-09 12:01:38.357082 Pytho...
    date_created:       2026-09-09 12:01:38.357096
    references:         doi.org/10.5194/gmd-19-5961-2026
    co2sys_parameters:  CO2System settings.\n├─ EQUILIBRIUM CONSTANTS:\n│  ├─...
    trace_parameters:   per_kg_sw_tf: True, canth_diseq: 1.0, eos: seawater, ...

>>> output.n2o_prime.data

array([0.75498545, 0.88762136])

```

## Arguments

Arguments to `trace_n2o` are the same as those of `trace`, with two optional additional parameters:

!!! inputs "Additional `trace_n2o` Arguments"

    * `atm_n2o_trajectory`: Integer between 1 and 9 specifying the atmospheric xN<sub>2</sub>O trajectory. This history has **not** been modified to reflect the values that would be expected in the surface ocean given the slow response of the surface ocean to a rapidly changing atmospheric value. Custom columns can be added to `data/N2OTrajectories.txt` and referenced here.
        * `1`. Historical/Linear **(default)**
        * `2`. SSP1_1.9
        * `3`. SSP1_2.6
        * `4`. SSP2_4.5
        * `5`. SSP3_7.0
        * `6`. SSP3_7.0_lowNTCF
        * `7`. SSP4_3.4
        * `8`. SSP4_6.0
        * `9`. SSP5_3.4_over

    * `preindustrial_xn2o`: Optional preindustrial reference xN<sub>2</sub>O value. The default is 271.