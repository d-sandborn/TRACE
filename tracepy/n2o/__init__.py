"""Plugin for preformed and transient nitrous oxide estimation using TRACE age estimates."""

import sys
import numpy as np
import numpy.typing as npt
from os.path import dirname, join as joinpath

import gsw
import seawater as sw
from numba import njit
from scipy.interpolate import interp1d

from tracepy import trace
from tracepy.utils import format_trace_output, inverse_gaussian_wrapper

DATADIR = joinpath(dirname(__file__), "data")


@njit
def n2o_eq(n2o_ppb, sal, temp):
    pt68 = temp * 1.00024
    y = pt68 + 273.15
    y_100 = y * 1e-2

    return (
        np.exp(
            -64.8539
            + 100.2520 * 100.0 / y
            + 25.2049 * np.log(y_100)
            + sal * (-0.062544 + y_100 * (0.035337 - 0.0054699 * y_100))
        )
        * n2o_ppb
    )


def trace_n2o(
    output_coordinates: npt.ArrayLike,
    dates: npt.ArrayLike,
    predictor_measurements: npt.ArrayLike,
    predictor_types: npt.ArrayLike,
    atm_co2_trajectory: int = 1,
    preindustrial_xco2: float = 280.0,
    preindustrial_xn2o: float = 271.0,
    output_filename: str = None,
    output_format: str = "xarray",
    delta_over_gamma: float = None,
    verbose_tf=True,
    error_codes: list = [-999, -9, -1e20],
    canth_diseq: float = 1.0,
    eos: str = "seawater",
    opt_pH_scale: int = 1,
    opt_k_carbonic: int = 10,  # LDK00
    opt_k_HSO4: int = 1,  # D90a
    opt_total_borate: int = 2,
    preformed_p: npt.ArrayLike = None,
    preformed_si: npt.ArrayLike = None,
    preformed_ta: npt.ArrayLike = None,
    scale_factors: npt.ArrayLike = None,
    meas_uncerts: npt.ArrayLike = None,
    per_kg_sw_tf: bool = True,
):
    """
    ▒▓████████▓▒░▒▓███████▓▒░C░▒▓██████▓▒░C░▒▓██████▓▒░░▒▓████████▓▒
    CC░▒▓█▓▒░CCC░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░CCCCCC
    CC░▒▓█▓▒░CCC░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░CCCCCC░▒▓█▓▒░CCCCCC
    CC░▒▓█▓▒░CCC░▒▓███████▓▒░░▒▓████████▓▒░▒▓█▓▒░CCCCCC░▒▓██████▓▒░C
    CC░▒▓█▓▒░CCC░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░CCCCCC░▒▓█▓▒░CCCCCC
    CC░▒▓█▓▒░CCC░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░CCCCCC
    CC░▒▓█▓▒░CCC░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓████████▓▒
    CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

                             Python v1.1.1

    Sandborn D. E., Carter, B. R., Barrett, R. 2026.
    https://doi.org/10.5194/gmd-19-5961-2026
    MATLAB - github.com/BRCScienceProducts/TRACEv1
    Python - github.com/d-sandborn/TRACE

    Generates etimates of ocean anthropogenic carbon and transient
    nitrous oxide content from user-supplied inputs of coordinates
    (lat, lon, depth), salinity, (optionally) temperature, and year.

    Information is also needed about the historical and/or future atm
    trajectory.  This information can be provided or default values can
    be asssumed. Missing data should be indicated with np.nan
    A nan coordinate will yield nan estimates for all equations at
    that coordinate. A nan parameter value will yield nan estimates for
    all equations that require that parameter. Please send questions or
    related requests to sandborn@uw.edu and brc.oceans@gmail.com.
    ==================================================================

    This module can be installed as an editable or installation or as a
    package in a virtual environment. It references its necessary data files
    (found in the TRACE/data directory) internally, but can output analysis
    results into the directory specified by output_filename using absolute
    or relative reference conventions.

    Parameters
    ----------
    output_coordinates : ArrayLike
        n by 3 array of coordinates (longitude decimal degrees E, latitude
        decimal degrees N, depth m) at which estimates are desired.
    dates : ArrayLike
        n by 1 array of years c.e. for which estimates are desired.
    predictor_measurements : ArrayLike
        n by y array of y parameter measurements (salinity, temperature)
        The column order (y columns) is specified by predictor_types.
        Temperature should be expressed as degrees C and salinity should be
        specified on the practical scale with the unitless convention.
        nan inputs are acceptable, but will lead to nan estimates for
        any equations that depend on that parameter. If temperature is not
        provided it will be estimated from salinity (not recommended).
    predictor_types : ArrayLike
        1 by y array indicating which
        parameter is in each column of 'predictor_measurements'.
        Note that salinity is required for all equations. This applies to all
        n estimates. Input parameter key:
            1. Salinity
            2. Temperature
    atm_co2_trajectory : int
        Integer between 1 and 9 specifying the atmospheric xCO2 trajectory:
            1. Historical/Linear
            2. SSP1_1.9
            3. SSP1_2.6
            4. SSP2_4.5
            5. SSP3_7.0
            6. SSP3_7.0_lowNTCF
            7. SSP4_3.4
            8. SSP4_6.0
            9. SSP5_3.4_over
        Custom columns can be added to the data/CO2ATrajectoreisAdjusted.txt
        file and referenced here.
    preindustrial_xco2 : float, optional
        Preindustrial reference xCO2 value. The default is 280.
    preindustrial_xn2o : float, optional
        Preindustrial reference xN2O value. The default is 271.
    output_filename: str, optional
        Filename for TRACE output to be saved in current working directory.
        If no filename is given, no file will be saved. Presently .nc, .csv,
        and .npy files can be saved, depending on the output_format. It is
        good practice (but not strictly required) to include the filetype
        (e.g. ".nc") as a part of this argument.
        The default is None.
    output_format: str, optional
        Object format for TRACE output, enabling work with xarray.datasets,
        pandas.dataframes, or numpy.arrays. Any of "xarray", "xr", "dataset",
        "netcdf", "nc", "pandas", "dataframe", "df", "csv", "numpy", "array",
        "matrix", "ndarray", or "np" are accepted.
        The default is 'xarray'
    verbose_tf : bool, optional
        Flag to control output verbosity. Setting this to False will
        make TRACE stop printing updates to the command line.  Warnings
        and errors, if any, will be given regardless.
        The default is True.
    error_codes : list, optional
        Error codes to be parsed as np.nan in input parameter arrays.
        The default is [-999, -9, -1e20].
    canth_diseq : int, optional
        Air-sea carbon dioxide equilibrium assumed for calculation of
        pCO2 as a function of atmospheric CO2. This should only be used if
        user-provided atmospheric trajectories not otherwise modified for
        anthropogenic carbon disequilibrium are being supplied.
        The default is 1.
    eos: str, optional
        Choice of seawater equation of state to use for temperature, density,
        and depth conversions. Available choices are 'seawater' (EOS-80)
        and 'gsw' (TEOS-10). 'seawater' will be deprecated, but is kept
        for compatibility with TRACEv1.
        The default is 'seawater'.
    delta_over_gamma: float, optional
        Ratio of second to first moments of inverse gaussian distribution
        used to convolute surface and interior histories of anthropogenic
        carbon.
        The default is 1.3038404810405297 to match TRACEv1 s.t.
        pf=makedist(‘InverseGaussian’,‘mu’,1,‘lambda’,3.4) is identical.
    opt_pH_scale: int, optional
        PyCO2SYS option for pH scale.
        The default is 1 (Total scale).
    opt_k_carbonic: int, optional
        PyCO2SYS option for carbonic acid dissociation constants.
        The default is 10 (LDK00).
    opt_k_HSO4: int, optional
        PyCO2SYS option for bisulfate dissociation constant.
        The default is 1 (D90a).
    opt_total_borate: int, optional
        PyCO2SYS option for borate:salinity relationship to use to estimate
        total borate.
        The default is 2.
    preformed_p: ArrayLike, optional
        n by 1 array of preformed P. When given along with preformed_ta and
        preformed_si, neural network estimation will be skipped.
        The default is None.
    preformed_si: ArrayLike, optional
        n by 1 array of preformed Si. When given along with preformed_ta and
        preformed_p, neural network estimation will be skipped.
        The default is None.
    preformed_ta: ArrayLike, optional
        n by 1 array of preformed TA. When given along with preformed_p and
        preformed_si, neural network estimation will be skipped.
        The default is None.
    scale_factors: ArrayLike, optional
        n by 1 array of scale factors for the inverse gaussian
        parameterization. When given neural network estimation will be skipped.
        The default is None.
    meas_uncerts : ArrayLike, optional
        ArrayLike object of measurement uncertainties presented in order
        indicated by 'predictor_types'. Providing these estimates may alter
        estimated uncertainties. Measurement uncertainties are a small part
        of TRACE estimate uncertainties for WOCE-quality measurements.
        However, estimate uncertainty scales with measurement uncertainty,
        so it is recommended that measurement uncertainties be specified
        for sensor measurements. If this optional input argument is not
        provided, the default WOCE-quality uncertainty is assumed.
        If values provided then the uncertainty estimates are assumed to
        apply uniformly to all input parameter measurements.
        The default is None.
    per_kg_sw_tf : bool, optional
        Retained for future development (allowing for flexible units
        for currently-unsupported predictors). The default is True.

    Returns
    -------
    output : xarray.Dataset
        CF-compliant ataset containing input parameters, estimated
        Canth, preformed properties, and associated metadata. This dataset
        is saved to the directory indicated by output_filename if provided.

    """

    output = trace(
        output_coordinates=output_coordinates,
        dates=dates,
        predictor_measurements=predictor_measurements,
        predictor_types=predictor_types,
        atm_co2_trajectory=atm_co2_trajectory,
        preindustrial_xco2=preindustrial_xco2,
        output_filename=None,
        output_format="xarray",
        delta_over_gamma=delta_over_gamma,
        verbose_tf=verbose_tf,
        error_codes=error_codes,
        canth_diseq=canth_diseq,
        eos=eos,
        opt_pH_scale=opt_pH_scale,
        opt_k_carbonic=opt_k_carbonic,
        opt_k_HSO4=opt_k_HSO4,
        opt_total_borate=opt_total_borate,
        preformed_p=preformed_p,
        preformed_si=preformed_si,
        preformed_ta=preformed_ta,
        scale_factors=scale_factors,
        meas_uncerts=meas_uncerts,
        per_kg_sw_tf=per_kg_sw_tf,
    )

    sfs = output["scale_factors"].values.astype(float)
    dates = np.asarray(dates, dtype=float).reshape(-1)
    delta_over_gamma = np.nanmedian(
        output["delta_over_gamma"].values.astype(float)
    )
    ventilation = inverse_gaussian_wrapper(
        x=np.arange(0.01, 5.01, 0.01), delta_over_gamma=delta_over_gamma
    )

    sal = output["salinity"].values.astype(float)
    temp = output["temperature"].values.astype(float)
    lon = output["lon"].values.astype(float)
    lat = output["lat"].values.astype(float)
    depth = output["depth"].values.astype(float)

    valid_indices = np.where(
        np.isfinite(sfs)
        & np.isfinite(dates)
        & np.isfinite(sal)
        & np.isfinite(temp)
        & np.isfinite(lon)
        & np.isfinite(lat)
        & np.isfinite(depth)
    )[0]

    n2o_rec = np.loadtxt(joinpath(DATADIR, "N2OTrajectories.txt"))
    n2o_rec = np.vstack([n2o_rec[0, :], n2o_rec])
    n2o_rec[0, 0] = -1e10

    n2o_set = interp1d(n2o_rec[:, 0], n2o_rec[:, atm_co2_trajectory])
    n2o_set = n2o_set(
        dates[valid_indices, None]
        - sfs[valid_indices, None] * np.arange(1, 501)
    ).dot(ventilation.T)

    if eos.lower() == "gsw":
        p = gsw.p_from_z(-depth, lat)
        sa = gsw.SA_from_SP(sal, p, lon, lat)
        temp = gsw.pt0_from_t(sa, temp, p)
    else:
        p = sw.pres(depth, lat)
        temp = sw.ptmp(sal, temp, p, 0)

    vpwp = np.exp(
        24.4543
        - 67.4509 * (100 / (293.15 + temp))
        - 4.8489 * np.log((293.15 + temp) / 100)
    )
    vpfac = 1 - vpwp * np.exp(-0.000544 * sal)

    pn2o_set = vpfac[valid_indices] * n2o_set
    pn2o_ref = vpfac[valid_indices] * preindustrial_xn2o
    n2o_out = n2o_eq(pn2o_set, sal[valid_indices], temp[valid_indices])
    n2o_ref = n2o_eq(pn2o_ref, sal[valid_indices], temp[valid_indices])

    full_n2o_set = np.full(len(sfs), np.nan)
    full_n2o_set[valid_indices] = n2o_set
    full_pn2o_set = np.full(len(sfs), np.nan)
    full_pn2o_set[valid_indices] = pn2o_set
    full_pn2o_ref = np.full(len(sfs), np.nan)
    full_pn2o_ref[valid_indices] = pn2o_ref
    full_n2o_out = np.full(len(sfs), np.nan)
    full_n2o_out[valid_indices] = n2o_out
    full_n2o_ref = np.full(len(sfs), np.nan)
    full_n2o_ref[valid_indices] = n2o_ref

    output["n2o"] = (
        ["loc"],
        full_n2o_out,
        {
            "units": "nmol kg-1",
            "long_name": "dissolved nitrous oxide",
            "standard_name": "moles_of_nitrous_oxide_per_unit_mass_in_sea_water",
        },
    )
    output["n2o_ref"] = (
        ["loc"],
        full_n2o_ref,
        {
            "units": "nmol kg-1",
            "long_name": "preindustrial dissolved nitrous oxide",
            "standard_name": "preindustrial_moles_of_nitrous_oxide_per_unit_mass_in_sea_water",
        },
    )
    output["pn2o"] = (
        ["loc"],
        full_pn2o_set,
        {
            "units": "natm",
            "long_name": "partial pressure of nitrous oxide",
            "standard_name": "partial_pressure_of_nitrous_oxide_in_sea_water",
        },
    )
    output["pn2o_ref"] = (
        ["loc"],
        full_pn2o_ref,
        {
            "units": "natm",
            "long_name": "preindustrial partial pressure of nitrous oxide",
            "standard_name": "preindustrial_partial_pressure_of_nitrous_oxide_in_sea_water",
        },
    )
    output["n2o_prime"] = (
        ["loc"],
        full_n2o_out - full_n2o_ref,
        {
            "units": "nmol kg-1",
            "long_name": "anthropogenic nitrous oxide",
        },
    )

    return format_trace_output(
        output,
        output_format=output_format,
        output_filename=output_filename,
    )
