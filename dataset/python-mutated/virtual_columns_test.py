from common import *

def test_add_virtual_columns_polar_velocities_to_cartesian():
    if False:
        while True:
            i = 10
    ds = vaex.example()
    ds.add_virtual_columns_cartesian_velocities_to_polar()
    ds.add_virtual_columns_cartesian_to_polar()
    ds.add_virtual_columns_polar_velocities_to_cartesian(vx_out='vx_', vy_out='vy_')
    np.testing.assert_almost_equal(ds.evaluate('vx'), ds.evaluate('vx_'), err_msg='error with converting polar to Cartesian velocities', decimal=3)
    np.testing.assert_almost_equal(ds.evaluate('vy'), ds.evaluate('vy_'), err_msg='error with converting polar to Cartesian velocities', decimal=3)
    ds.add_virtual_columns_polar_velocities_to_cartesian(azimuth='phi_polar', vx_out='vx_', vy_out='vy_')
    np.testing.assert_almost_equal(ds.evaluate('vx'), ds.evaluate('vx_'), err_msg='error with converting polar to Cartesian velocities', decimal=3)
    np.testing.assert_almost_equal(ds.evaluate('vy'), ds.evaluate('vy_'), err_msg='error with converting polar to Cartesian velocities', decimal=3)
    ds.add_virtual_columns_cartesian_angular_momenta(Lx='Lx_', Ly='Ly_', Lz='Lz_')
    ds['L_'] = np.sqrt(ds.Lx_ ** 2.0 + ds.Ly_ ** 2.0 + ds.Lz_ ** 2.0)
    np.testing.assert_almost_equal(ds.Lz.values, ds.Lz_.values, err_msg='error when calculating Lz', decimal=3)
    np.testing.assert_almost_equal(ds.L.values, ds.L_.values, err_msg='error when calculating the Ltotal', decimal=3)

def test_add_and_delete_virtual_column():
    if False:
        for i in range(10):
            print('nop')
    ds = vaex.example()
    ds.add_virtual_column('double_x', 'x * 2')
    assert 'double_x' in ds.get_column_names()
    ds.delete_virtual_column('double_x')
    assert 'double_x' not in ds.get_column_names()