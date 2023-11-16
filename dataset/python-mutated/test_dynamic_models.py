import torch
from pyro.contrib.tracking.dynamic_models import NcpContinuous, NcpDiscrete, NcvContinuous, NcvDiscrete
from tests.common import assert_equal, assert_not_equal

def assert_cov_validity(cov, eigenvalue_lbnd=0.0, condition_number_ubnd=1000000.0):
    if False:
        print('Hello World!')
    '\n    cov: covariance matrix\n    eigenvalue_lbnd: eigenvalues should be at least this much greater than\n      zero. Must be strictly positive.\n    condition_number_ubnd: inclusive upper bound on matrix condition\n      number. Must be greater or equal to 1.0.\n    '
    assert eigenvalue_lbnd >= 0.0, 'Covariance eigenvalue lower bound must be > 0.0!'
    assert condition_number_ubnd >= 1.0, 'Covariance condition number bound must be >= 1.0!'
    assert (cov.t() == cov).all(), 'Covariance must be symmetric!'
    ws = torch.linalg.eigvalsh(cov)
    w_min = torch.min(ws)
    w_max = torch.max(ws)
    assert w_min > 0.0, 'Covariance must be strictly positive!'
    assert w_min >= eigenvalue_lbnd, 'Covariance eigenvalues must be >= lower bound!'
    assert w_max / w_min <= condition_number_ubnd, 'Condition number must be <= upper bound!'

def test_NcpContinuous():
    if False:
        return 10
    framerate = 100
    dt = 1.0 / framerate
    d = 3
    ncp = NcpContinuous(dimension=d, sv2=2.0)
    assert ncp.dimension == d
    assert ncp.dimension_pv == 2 * d
    assert ncp.num_process_noise_parameters == 1
    x = torch.rand(d)
    y = ncp(x, dt)
    assert_equal(y, x)
    dx = ncp.geodesic_difference(x, y)
    assert_equal(dx, torch.zeros(d))
    x_pv = ncp.mean2pv(x)
    assert len(x_pv) == 6
    assert_equal(x, x_pv[:d])
    assert_equal(torch.zeros(d), x_pv[d:])
    P = torch.eye(d)
    P_pv = ncp.cov2pv(P)
    assert P_pv.shape == (2 * d, 2 * d)
    P_pv_ref = torch.zeros((2 * d, 2 * d))
    P_pv_ref[:d, :d] = P
    assert_equal(P_pv_ref, P_pv)
    Q = ncp.process_noise_cov(dt)
    Q1 = ncp.process_noise_cov(dt)
    assert_equal(Q, Q1)
    assert Q1.shape == (d, d)
    assert_cov_validity(Q1)
    dx = ncp.process_noise_dist(dt).sample()
    assert dx.shape == (ncp.dimension,)

def test_NcvContinuous():
    if False:
        for i in range(10):
            print('nop')
    framerate = 100
    dt = 1.0 / framerate
    d = 6
    ncv = NcvContinuous(dimension=d, sa2=2.0)
    assert ncv.dimension == d
    assert ncv.dimension_pv == d
    assert ncv.num_process_noise_parameters == 1
    x = torch.rand(d)
    y = ncv(x, dt)
    assert_equal(y[0], x[0] + dt * x[d // 2])
    dx = ncv.geodesic_difference(x, y)
    assert_not_equal(dx, torch.zeros(d))
    x_pv = ncv.mean2pv(x)
    assert len(x_pv) == d
    assert_equal(x, x_pv)
    P = torch.eye(d)
    P_pv = ncv.cov2pv(P)
    assert P_pv.shape == (d, d)
    assert_equal(P, P_pv)
    Q = ncv.process_noise_cov(dt)
    Q1 = ncv.process_noise_cov(dt)
    assert_equal(Q, Q1)
    assert Q1.shape == (d, d)
    assert_cov_validity(Q1)
    dx = ncv.process_noise_dist(dt).sample()
    assert dx.shape == (ncv.dimension,)

def test_NcpDiscrete():
    if False:
        return 10
    framerate = 100
    dt = 1.0 / framerate
    d = 3
    ncp = NcpDiscrete(dimension=d, sv2=2.0)
    assert ncp.dimension == d
    assert ncp.dimension_pv == 2 * d
    assert ncp.num_process_noise_parameters == 1
    x = torch.rand(d)
    y = ncp(x, dt)
    assert_equal(y, x)
    dx = ncp.geodesic_difference(x, y)
    assert_equal(dx, torch.zeros(d))
    x_pv = ncp.mean2pv(x)
    assert len(x_pv) == 6
    assert_equal(x, x_pv[:d])
    assert_equal(torch.zeros(d), x_pv[d:])
    P = torch.eye(d)
    P_pv = ncp.cov2pv(P)
    assert P_pv.shape == (2 * d, 2 * d)
    P_pv_ref = torch.zeros((2 * d, 2 * d))
    P_pv_ref[:d, :d] = P
    assert_equal(P_pv_ref, P_pv)
    Q = ncp.process_noise_cov(dt)
    Q1 = ncp.process_noise_cov(dt)
    assert_equal(Q, Q1)
    assert Q1.shape == (d, d)
    assert_cov_validity(Q1)
    dx = ncp.process_noise_dist(dt).sample()
    assert dx.shape == (ncp.dimension,)

def test_NcvDiscrete():
    if False:
        while True:
            i = 10
    framerate = 100
    dt = 1.0 / framerate
    dt = 100
    d = 6
    ncv = NcvDiscrete(dimension=d, sa2=2.0)
    assert ncv.dimension == d
    assert ncv.dimension_pv == d
    assert ncv.num_process_noise_parameters == 1
    x = torch.rand(d)
    y = ncv(x, dt)
    assert_equal(y[0], x[0] + dt * x[d // 2])
    dx = ncv.geodesic_difference(x, y)
    assert_not_equal(dx, torch.zeros(d))
    x_pv = ncv.mean2pv(x)
    assert len(x_pv) == d
    assert_equal(x, x_pv)
    P = torch.eye(d)
    P_pv = ncv.cov2pv(P)
    assert P_pv.shape == (d, d)
    assert_equal(P, P_pv)
    Q = ncv.process_noise_cov(dt)
    Q1 = ncv.process_noise_cov(dt)
    assert_equal(Q, Q1)
    assert Q1.shape == (d, d)