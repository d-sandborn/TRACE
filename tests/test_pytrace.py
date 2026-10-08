import pytest
import numpy as np
from tracepy import trace, integrate_column, trace_n2o


def test_dummy():
    """Are internal tests working?"""
    assert 1 == 1


def test_trace_matlab():
    """Is TRACE giving identical results to the TRACEv1 check values?"""
    output = trace(
        output_coordinates=np.array([[0, 0, 0], [0, 0, 0]]),
        dates=np.array([2000, 2200]),
        predictor_measurements=np.array([[35, 20], [35, 20]]),
        predictor_types=np.array([1, 2]),
        atm_co2_trajectory=9,
    )
    assert isinstance(output.canth.data[0], float)
    assert isinstance(output.canth.data[1], float)
    assert np.isclose(output.canth.data[0], 47.1840234333929)
    assert np.isclose(output.canth.data[1], 78.8042113781469)
    # assert np.abs(output.canth.data[0] - 47.1840234333929) < 0.00001
    # assert np.abs(output.canth.data[1] - 78.8042113781469) < 0.00001


def test_trace_matlab_no_temperature():
    """Is TRACE giving identical results to no-T TRACEv1 check values?"""
    output = trace(
        output_coordinates=np.array([[0, 0, 0], [0, 0, 0]]),
        dates=np.array([2000, 2010]),
        predictor_measurements=np.array([[35], [35]]),
        predictor_types=np.array([1]),
        atm_co2_trajectory=1,
    )
    assert isinstance(output.canth.data[0], float)
    assert isinstance(output.canth.data[1], float)
    assert np.isclose(output.canth.data[0], 55.3202019474529)
    assert np.isclose(output.canth.data[1], 65.5560979572231)
    # assert np.abs(output.canth.data[0] - 55.3202019474529) < 0.00001
    # assert np.abs(output.canth.data[1] - 65.5560979572231) < 0.00001


def test_trace_n2o():
    """Is TRACE giving stable results for canth and n2o'?"""
    output = trace_n2o(
        output_coordinates=np.array([[0, 0, 0], [0, 0, 0]]),
        dates=np.array([2000, 2200]),
        predictor_measurements=np.array([[35, 20], [35, 20]]),
        predictor_types=np.array([1, 2]),
        atm_n2o_trajectory=1,
    )
    assert isinstance(output.canth.data[0], float)
    assert isinstance(output.canth.data[1], float)
    assert isinstance(output.n2o_prime.data[0], float)
    assert isinstance(output.n2o_prime.data[1], float)
    assert np.isclose(output.canth.data[0], 47.49327014256119)
    assert np.isclose(output.canth.data[1], 195.60756515621028)
    assert np.isclose(output.n2o_prime.data[0], 0.8937910007428256)
    assert np.isclose(output.n2o_prime.data[1], 4.911602067590993)


def test_integrate_column():
    """Is the integration routine interpolating correctly?"""
    integral = integrate_column(
        integrand=np.array([1, 2, 3]),
        salinity=np.array([35, 35, 35]),
        temperature=np.array([4, 4, 4]),
        depth=np.array([100, 200, 300]),
        lat=0,
        bottom=250,
    )
    assert integral - 321347.0731205583 < 0.00001
