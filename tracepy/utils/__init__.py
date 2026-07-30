"""Utility functions."""

import numpy as np
import warnings
import datetime
from scipy.stats import invgauss
from gsw import rho_t_exact, p_from_z
from scipy.interpolate import PchipInterpolator
from scipy.integrate import romb
import numpy.typing as npt

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
# from seawater import ptmp, dens, pres
from shapely.geometry.polygon import Polygon
import geopandas as gpd


def equation_check(equation):
    """Check equation inputs and assigns them to be [1] regardless.

    Largely a carry-over from ESPER, which accepted more options."""
    match equation:
        case []:
            equation = [1]
        case [0]:
            equation = [1]
        case [1]:
            equation = [1]
        case _:
            warnings.warn(
                "Input 'equations' could not be parsed. Setting to [1]."
            )
            equation = [1]
    return equation


def units_check(per_kg_sw_tf):
    """Check for per_kg_sw_tf input and setting default if not given.

    This input is not needed for TRACE, currently."""
    if not per_kg_sw_tf:
        warnings.warn(
            "Optional argument per_kg_sw_tf is not in use. Setting to True."
        )
        per_kg_sw_tf = True
    return per_kg_sw_tf


def preindustrial_check(preindustrial_xco2):
    """Check for preindustrial_xco2 input and setting default if not given."""
    if not isinstance(preindustrial_xco2, float) and not isinstance(
        preindustrial_xco2, int
    ):
        warnings.warn(
            "Preindustrial_xco2 could not be parsed as int or float. Setting to 280."
        )
        preindustrial_xco2 = 280
    return preindustrial_xco2


def uncerts_check(meas_uncerts, predictor_measurements, predictor_types):
    """
    Checks the meas_uncerts argument.

    This also deals with the
    possibility that the user has provided a single set of uncertainties
    for all estimates. Also coerces arrays to np.array.
    """
    if meas_uncerts is not None:
        # Copying uncertainty estimates for all estimates if only singular
        # values are provided.
        use_default_uncertainties = False
        input_u = np.ones(len(predictor_measurements)) * meas_uncerts
    else:
        use_default_uncertainties = True
        input_u = None
        try:
            predictor_measurements = np.asarray(predictor_measurements)
            predictor_types = np.asarray(predictor_types)
        except Exception as e:
            print(
                f"{e}\nCould not convert at least one of predictor_measurements and/or predictor_types to a numpy array."
            )

    return (
        meas_uncerts,
        input_u,
        use_default_uncertainties,
        predictor_measurements,
        predictor_types,
    )


def depth_check(output_coordinates, valid_indices):
    """
    This step checks for negative depths.

    If found, it changes them to
    positive depths and issues a warning. Also coerces arrays to np.array.
    """
    # try:
    #     output_coordinates = np.asarray(output_coordinates)
    # except Exception as e:
    #     print(f"{e}\nCould not convert output_coordinates to a numpy array.")
    if np.any(output_coordinates[valid_indices, 2] < 0):
        warnings.warn(
            "Negative depths were detected and changed to positive values."
        )
        output_coordinates[valid_indices, 2] = np.abs(
            output_coordinates[valid_indices, 2]
        )
    return output_coordinates


def coordinate_check(output_coordinates, valid_indices):
    """Book-keeping coordinate inputs and adjusting negative longitudes."""
    if np.any(np.abs(output_coordinates[:, 1]) > 90):
        raise ValueError(
            "A latitude >90 degrees (N or S) has been detected.  Verify latitude is in the 2nd colum of the coordinate input."
        )
    C = output_coordinates[valid_indices, :].copy()
    C[:, 0] = np.mod(C[:, 0], 360)
    C[C[:, 0] < 0, 0] += 360
    return output_coordinates, C


def prepare_uncertainties(
    predictor_measurements,
    predictor_types,
    valid_indices,
    meas_uncerts=None,
):
    predictor_measurements = np.asarray(predictor_measurements, dtype=float)
    predictor_types = np.asarray(predictor_types, dtype=int)
    valid_indices = np.asarray(valid_indices, dtype=int)

    if predictor_measurements.ndim != 2:
        raise ValueError("predictor_measurements must have shape (n, p).")
    if predictor_types.ndim != 1:
        raise ValueError("predictor_types must have shape (p,).")
    if predictor_measurements.shape[1] != predictor_types.size:
        raise ValueError(
            "predictor_measurements and predictor_types must have matching predictor columns."
        )
    if np.any(~np.isin(predictor_types, [1, 2])):
        raise ValueError(
            "prepare_uncertainties currently supports only predictor_types 1 and 2."
        )
    if np.any(valid_indices < 0) or np.any(
        valid_indices >= predictor_measurements.shape[0]
    ):
        raise IndexError("valid_indices contains an out-of-range row index.")

    predictor_columns = predictor_types - 1
    default_u_full = np.zeros(
        (predictor_measurements.shape[0], 6), dtype=float
    )
    default_u_full[:, predictor_columns] = predictor_measurements

    salinity_columns = predictor_columns[predictor_types == 1]
    temperature_columns = predictor_columns[predictor_types == 2]

    if salinity_columns.size:
        default_u_full[:, salinity_columns] = 0.003
    if temperature_columns.size:
        default_u_full[:, temperature_columns] = 0.003

    input_u_full = default_u_full.copy()

    if meas_uncerts is not None:
        meas_uncerts = np.asarray(meas_uncerts, dtype=float)
        if np.any(~np.isfinite(meas_uncerts)):
            raise ValueError(
                "meas_uncerts cannot contain NaN or infinite values."
            )
        if np.any(meas_uncerts < 0):
            raise ValueError("meas_uncerts cannot contain negative values.")
        if meas_uncerts.ndim == 1:
            if meas_uncerts.size != predictor_types.size:
                raise ValueError(
                    "1-D meas_uncerts must have one value per predictor_type."
                )
            input_u_full[:, predictor_columns] = meas_uncerts.reshape(1, -1)
        elif meas_uncerts.ndim == 2:
            if meas_uncerts.shape != predictor_measurements.shape:
                raise ValueError(
                    "2-D meas_uncerts must have the same shape as predictor_measurements."
                )
            input_u_full[:, predictor_columns] = meas_uncerts
        else:
            raise ValueError(
                "meas_uncerts must be None, 1-D, or 2-D after normalization."
            )
        input_u_full = np.maximum(input_u_full, default_u_full)

    return default_u_full[valid_indices, :], input_u_full[valid_indices, :]


def inverse_gaussian_wrapper(x, delta_over_gamma=1.3038404810405297):
    """
    Calculate ventilation distributions (assumed probability
    distribution).

    lambda should perhaps be 1/1.3 from He et al.
    Note that invgauss calls are different in TRACE-Python and TRACEv1!
    Also note that TRACE approximates mu as 3.4 instead of ~3.38,
    leading to the default delta_over_gamma = sqrt(3.4/2).
    """
    nu = 2 * (delta_over_gamma) ** 2  # default 3.4
    lam = 1
    y = invgauss.pdf(x, mu=nu / lam, scale=lam, loc=0)
    y = y / y.sum()
    return y


def inpolygon(xq, yq, xv, yv):
    """Test for points in polygon."""
    polygon_geom = Polygon(zip(xv, yv))
    polygon = gpd.GeoDataFrame(
        index=[0], crs="epsg:4326", geometry=[polygon_geom]
    )
    geo = gpd.points_from_xy(xq, yq)
    points = gpd.GeoDataFrame(geometry=geo, crs=polygon.crs)
    pointInPolys = points.intersects(polygon.union_all())
    return pointInPolys


# try: #this mistakenly identifies points within convex boundaries
#    points = np.array([xq, yq]).T
#    path = np.array([xv, yv]).T
#    tri = Delaunay(path)
#    return tri.find_simplex(points) >= 0
# except:
#    return np.array([False] * len(xq))


def say_hello():
    """It's only polite."""
    print("""
▒▓████████▓▒░▒▓███████▓▒░C░▒▓██████▓▒░C░▒▓██████▓▒░░▒▓████████▓▒
CC░▒▓█▓▒░CCC░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░CCCCCC
CC░▒▓█▓▒░CCC░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░CCCCCC░▒▓█▓▒░CCCCCC
CC░▒▓█▓▒░CCC░▒▓███████▓▒░░▒▓████████▓▒░▒▓█▓▒░CCCCCC░▒▓██████▓▒░C
CC░▒▓█▓▒░CCC░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░CCCCCC░▒▓█▓▒░CCCCCC
CC░▒▓█▓▒░CCC░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░CCCCCC
CC░▒▓█▓▒░CCC░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓████████▓▒
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

                         Python v1.0.0

Sandborn D. E., Carter, B. R., Barrett, R. 2025.
https://doi.org/10.5194/essd-17-3073-2025
MATLAB - github.com/BRCScienceProducts/TRACEv1
Python - github.com/d-sandborn/pyTRACE""")


def decimal_year_to_iso_timestamp(  # for CF Conventions
    decimal_year_input: np.ndarray | float,
) -> np.ndarray | str:
    def _convert_single_decimal_year(decimal_year: float) -> str:
        if np.isnan(decimal_year):
            return "NaT"
        year = int(decimal_year)
        fraction = decimal_year - year

        days_in_year = (
            366
            if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
            else 365
        )

        total_seconds_in_year = days_in_year * 24 * 60 * 60
        offset_seconds = fraction * total_seconds_in_year

        start_of_year_utc = datetime.datetime(
            year, 1, 1, 0, 0, 0, 0, tzinfo=datetime.timezone.utc
        )

        dt_object_utc = start_of_year_utc + datetime.timedelta(
            seconds=offset_seconds
        )

        iso_timestamp = dt_object_utc.isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        )
        return iso_timestamp

    if isinstance(decimal_year_input, np.ndarray):
        vectorized_converter = np.vectorize(
            _convert_single_decimal_year, otypes=[str]
        )
        return vectorized_converter(decimal_year_input)
    else:
        return _convert_single_decimal_year(decimal_year_input)


def _integrate_column(
    integrand,
    salinity,
    temperature,
    depth,
    lat: float,
    bottom: float,
    top: float = 0,
    romb_resolution: int = 10,
):
    shapes = {v.shape for v in [integrand, salinity, temperature, depth]}
    if not len(shapes) == 1:
        raise ValueError("The shapes of the input vectors do not match.")
    num_target_points_for_romb = (2**romb_resolution) + 1
    # depthgrid, latgrid = np.meshgrid(ds.depth.data, ds.lat.data)
    pressure = p_from_z(-depth, lat * np.ones(len(depth)))
    profile = integrand * rho_t_exact(
        salinity, temperature, pressure
    )  # micromol/kg to micromol/m^3

    valid_indices = np.logical_and(
        (depth <= bottom),
        (~np.isnan(profile)),
    )

    if np.sum(valid_indices) < 1:  # nada if no water
        raise ValueError(
            "No valid indices to integrate. Check that at least one depth is less than the bottom depth, and that other values are reasonable."
        )
    elif np.sum(valid_indices) < 2:  # simple average if only one block
        warnings.warn(
            "Only one valid index to integrate! Assuming that this is the average value throughout the column. "
        )
        column_inventory = np.nansum(profile * (bottom - top))
    elif np.sum(valid_indices) >= 2:  # pchip/romb
        valid_original_depths = depth[valid_indices]
        valid_profile = profile[valid_indices]

        pchip_interpolator = PchipInterpolator(
            valid_original_depths,
            valid_profile,
            extrapolate=True,
        )
        dynamic_target_depth_points = np.linspace(
            top, bottom, num_target_points_for_romb
        )
        h = dynamic_target_depth_points[1] - dynamic_target_depth_points[0]
        # Perform interpolation
        interpolated_values = pchip_interpolator(dynamic_target_depth_points)
        try:
            column_inventory = romb(interpolated_values, dx=h)
        except ValueError as e:
            print(f"Error during Romberg integration:\n{e}")

    return column_inventory


def _dataset_to_single_variable(obj, *, name: str):
    if hasattr(obj, "data_vars"):
        data_vars = list(obj.data_vars)
        if len(data_vars) != 1:
            raise ValueError(f"{name} must contain exactly one data variable.")
        return obj[data_vars[0]]
    return obj


def _object_missing_to_nan(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != object:
        return arr
    try:
        import pandas as pd

        return np.where(pd.isna(arr), np.nan, arr)
    except Exception:
        values = []
        for value in arr.ravel():
            missing = value is None or value is np.ma.masked
            if not missing:
                try:
                    missing = bool(np.isnan(value))
                except Exception:
                    missing = value.__class__.__name__ in {"NAType", "NaTType"}
            values.append(np.nan if missing else value)
        return np.asarray(values, dtype=object).reshape(arr.shape)


def _as_numpy(obj, *, name: str, dtype=float) -> np.ndarray:
    if obj is None:
        raise ValueError(f"{name} cannot be None.")

    obj = _dataset_to_single_variable(obj, name=name)

    if hasattr(obj, "to_numpy"):
        try:
            obj = obj.to_numpy(dtype=dtype, na_value=np.nan)
        except TypeError:
            try:
                obj = obj.to_numpy(dtype=dtype)
            except TypeError:
                obj = obj.to_numpy()
    elif hasattr(obj, "values"):
        obj = obj.values

    arr = np.asanyarray(obj)

    if np.ma.isMaskedArray(arr):
        arr = np.ma.filled(arr.astype(float), np.nan)

    arr = _object_missing_to_nan(np.asarray(arr))

    try:
        arr = arr.astype(dtype, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be numeric or convertible to numeric."
        ) from exc

    return np.asarray(arr)


def _as_predictor_types(obj) -> np.ndarray:
    arr = _as_numpy(obj, name="predictor_types", dtype=float).ravel()

    if arr.size == 0:
        raise ValueError(
            "predictor_types must contain at least one predictor code."
        )
    if np.any(~np.isfinite(arr)):
        raise ValueError(
            "predictor_types cannot contain NaN or infinite values."
        )
    if not np.all(arr == np.floor(arr)):
        raise ValueError(
            "predictor_types must contain integer predictor codes."
        )

    arr = arr.astype(int, copy=False)

    if not np.all(np.isin(arr, [1, 2])):
        raise ValueError(
            "predictor_types currently supports only 1=salinity and 2=temperature."
        )
    if np.unique(arr).size != arr.size:
        raise ValueError("predictor_types contains duplicate predictor codes.")
    if 1 not in arr:
        raise ValueError("Salinity predictor_type 1 is required for TRACE.")

    return arr


def _broadcast_or_validate_n(
    arr: np.ndarray, *, n: int, name: str
) -> np.ndarray:
    arr = np.ravel(arr)

    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    if arr.size == n:
        return arr
    if arr.size == 1:
        return np.full(n, arr.item(), dtype=arr.dtype)

    raise ValueError(f"{name} has length {arr.size}, but expected length {n}.")


def _normalize_optional_vector(obj, *, n: int, name: str):
    if obj is None:
        return None

    arr = _as_numpy(obj, name=name, dtype=float)

    if arr.size == 0:
        return None

    return _broadcast_or_validate_n(arr, n=n, name=name)


def _normalize_output_coordinates(obj) -> np.ndarray:
    arr = _as_numpy(obj, name="output_coordinates", dtype=float)

    if arr.ndim == 0:
        raise ValueError(
            "output_coordinates must contain longitude, latitude, and depth."
        )
    if arr.ndim == 1:
        if arr.size != 3:
            raise ValueError(
                "A 1-D output_coordinates input must have exactly 3 values: lon, lat, depth."
            )
        arr = arr.reshape(1, 3)
    elif arr.ndim == 2:
        if arr.shape[0] == 3 and arr.shape[1] != 3:
            warnings.warn(
                "output_coordinates appears to be transposed; converting shape (3, n) to (n, 3)."
            )
            arr = arr.T
        if arr.shape[1] != 3:
            raise ValueError(
                "output_coordinates must have shape (n, 3), with columns lon, lat, depth."
            )
    else:
        raise ValueError("output_coordinates must be 1-D or 2-D.")

    if arr.shape[0] == 0:
        raise ValueError(
            "output_coordinates must contain at least one location."
        )

    return arr


def _normalize_predictor_measurements(obj, *, n: int, p: int) -> np.ndarray:
    arr = _as_numpy(obj, name="predictor_measurements", dtype=float)

    if arr.ndim == 0:
        if p != 1:
            raise ValueError(
                "Scalar predictor_measurements is only valid when predictor_types has one entry."
            )
        return np.full((n, 1), arr.item(), dtype=float)

    if arr.ndim == 1:
        if n == 1 and arr.size == p:
            return arr.reshape(1, p)
        if p == 1 and arr.size == n:
            return arr.reshape(n, 1)
        if arr.size == 1 and p == 1:
            return np.full((n, 1), arr.item(), dtype=float)
        raise ValueError(
            "Ambiguous 1-D predictor_measurements. For one location use length p; for one predictor across n locations use length n."
        )

    if arr.ndim == 2:
        if arr.shape == (p, n) and arr.shape != (n, p):
            warnings.warn(
                "predictor_measurements appears to be transposed; converting shape (p, n) to (n, p)."
            )
            arr = arr.T
        if arr.shape != (n, p):
            raise ValueError(
                f"predictor_measurements has shape {arr.shape}, but expected {(n, p)}."
            )
        return arr

    raise ValueError("predictor_measurements must be scalar, 1-D, or 2-D.")


def _normalize_meas_uncerts(obj, *, n: int, p: int):
    if obj is None:
        return None

    arr = _as_numpy(obj, name="meas_uncerts", dtype=float)

    if arr.size == 0:
        return None

    if arr.ndim == 0:
        arr = np.full((p,), arr.item(), dtype=float)
    elif arr.ndim == 1:
        if arr.size == p:
            pass
        elif arr.size == n and p == 1:
            arr = arr.reshape(n, 1)
        elif arr.size == 1:
            arr = np.full((p,), arr.item(), dtype=float)
        else:
            raise ValueError(
                "meas_uncerts must be scalar, length p, length n for one predictor, or shape (n, p)."
            )
    elif arr.ndim == 2:
        if arr.shape == (p, n) and arr.shape != (n, p):
            warnings.warn(
                "meas_uncerts appears to be transposed; converting shape (p, n) to (n, p)."
            )
            arr = arr.T
        if arr.shape != (n, p):
            raise ValueError(
                f"meas_uncerts has shape {arr.shape}, expected {(n, p)}."
            )
    else:
        raise ValueError("meas_uncerts must be scalar, 1-D, or 2-D.")

    if np.any(~np.isfinite(arr)):
        raise ValueError("meas_uncerts cannot contain NaN or infinite values.")
    if np.any(arr < 0):
        raise ValueError("meas_uncerts cannot contain negative values.")

    return arr


def normalize_trace_inputs(
    output_coordinates: npt.ArrayLike,
    dates: npt.ArrayLike,
    predictor_measurements: npt.ArrayLike,
    predictor_types: npt.ArrayLike,
    meas_uncerts: npt.ArrayLike | None = None,
    preformed_p: npt.ArrayLike | None = None,
    preformed_si: npt.ArrayLike | None = None,
    preformed_ta: npt.ArrayLike | None = None,
    scale_factors: npt.ArrayLike | None = None,
):
    """Makes sure everything is a np array of the correct dims."""
    predictor_types = _as_predictor_types(predictor_types)
    output_coordinates = _normalize_output_coordinates(output_coordinates)
    n = output_coordinates.shape[0]  # same as the dims in TRACE-MATLAB
    p = predictor_types.size

    dates = _broadcast_or_validate_n(
        _as_numpy(dates, name="dates", dtype=float), n=n, name="dates"
    )
    predictor_measurements = _normalize_predictor_measurements(
        predictor_measurements, n=n, p=p
    )
    meas_uncerts = _normalize_meas_uncerts(meas_uncerts, n=n, p=p)
    preformed_p = _normalize_optional_vector(
        preformed_p, n=n, name="preformed_p"
    )
    preformed_si = _normalize_optional_vector(
        preformed_si, n=n, name="preformed_si"
    )
    preformed_ta = _normalize_optional_vector(
        preformed_ta, n=n, name="preformed_ta"
    )
    scale_factors = _normalize_optional_vector(
        scale_factors, n=n, name="scale_factors"
    )

    return {
        "output_coordinates": output_coordinates,
        "dates": dates,
        "predictor_measurements": predictor_measurements,
        "predictor_types": predictor_types,
        "meas_uncerts": meas_uncerts,
        "preformed_p": preformed_p,
        "preformed_si": preformed_si,
        "preformed_ta": preformed_ta,
        "scale_factors": scale_factors,
    }


def valid_trace_indices(
    output_coordinates: np.ndarray,
    dates: np.ndarray,
    predictor_measurements: np.ndarray,
    predictor_types: np.ndarray,
) -> np.ndarray:
    output_coordinates = np.asarray(output_coordinates)
    dates = np.asarray(dates)
    predictor_measurements = np.asarray(predictor_measurements)
    predictor_types = np.asarray(predictor_types)

    if output_coordinates.ndim != 2 or output_coordinates.shape[1] != 3:
        raise ValueError("output_coordinates must have shape (n, 3).")
    if dates.ndim != 1:
        raise ValueError("dates must have shape (n,).")
    if predictor_measurements.ndim != 2:
        raise ValueError("predictor_measurements must have shape (n, p).")
    if predictor_types.ndim != 1:
        raise ValueError("predictor_types must have shape (p,).")
    if output_coordinates.shape[0] != dates.shape[0]:
        raise ValueError(
            "output_coordinates and dates must have the same number of rows."
        )
    if output_coordinates.shape[0] != predictor_measurements.shape[0]:
        raise ValueError(
            "output_coordinates and predictor_measurements must have the same number of rows."
        )
    if predictor_measurements.shape[1] != predictor_types.size:
        raise ValueError(
            "predictor_measurements and predictor_types must have the same number of predictor columns."
        )

    salinity_columns = np.flatnonzero(predictor_types == 1)

    if salinity_columns.size != 1:
        raise ValueError(
            "Exactly one salinity predictor_type 1 column is required for TRACE."
        )

    coord_missing = ~np.isfinite(output_coordinates).all(axis=1)
    date_missing = ~np.isfinite(dates)
    salinity_missing = ~np.isfinite(
        predictor_measurements[:, salinity_columns[0]]
    )
    valid_mask = ~(coord_missing | date_missing | salinity_missing)

    return np.flatnonzero(valid_mask)  # valid 1D vector


def format_trace_output(output, output_format="xarray", output_filename=None):
    fmt = str(output_format).strip().lower()
    result = output
    save_method = None

    if fmt in {"xarray", "xr", "dataset", "netcdf", "nc"}:
        save_method = "netcdf"
    elif fmt in {"pandas", "dataframe", "df", "csv"}:
        result = output.to_dataframe()
        save_method = "csv"
    elif fmt in {"numpy", "array", "matrix", "ndarray", "np"}:
        result = output.to_dataframe().to_numpy()
        save_method = "npy"
    else:
        raise ValueError(
            "output_format must be one of 'xarray', 'pandas', or 'numpy'."
        )

    if output_filename is not None:
        try:
            if save_method == "netcdf":
                result.to_netcdf(output_filename)
            elif save_method == "csv":
                result.to_csv(output_filename)
            elif save_method == "npy":
                np.save(output_filename, result)
        except Exception as exc:
            print("File " + output_filename + " could not be saved")
            print(exc)

    return result
