__author__ = 'saeedamen'

class MarketConstants(object):
    """Has various constants required for the finmarketpy project. These have been defined as static variables.

    """
    import platform
    plat = str(platform.platform()).lower()
    if 'linux' in plat:
        generic_plat = 'linux'
    elif 'windows' in plat:
        generic_plat = 'windows'
    elif 'darwin' in plat or 'macos' in plat:
        generic_plat = 'mac'
    backtest_thread_technique = 'multiprocessing'
    multiprocessing_library = 'multiprocess'
    backtest_thread_no = {'linux': 8, 'windows': 1, 'mac': 8}
    hdf5_file_econ_file = 'x'
    db_database_econ_file = ''
    db_server = '127.0.0.1'
    db_port = '27017'
    db_username = 'admin_root'
    db_password = 'TOFILL'
    write_engine = 'arctic'
    spot_depo_tenor = 'ON'
    currencies_with_365_basis = ['AUD', 'CAD', 'GBP', 'NZD']
    output_calculation_fields = False
    fx_forwards_points_divisor_1 = ['IDR']
    fx_forwards_points_divisor_100 = ['JPY']
    fx_forwards_points_divisor_1000 = []
    fx_forwards_tenor_for_interpolation = ['1W', '2W', '3W', '1M', '2M', '3M', '4M', '6M', '9M', '1Y', '2Y', '3Y', '5Y']
    fx_forwards_trading_tenor = '1M'
    fx_forwards_cum_index = 'mult'
    fx_forwards_roll_event = 'month-end'
    fx_forwards_roll_days_before = 5
    fx_forwards_roll_months = 1
    fx_options_points_divisor_100 = ['JPY']
    fx_options_points_divisor_1000 = []
    fx_options_tenor_for_interpolation = ['ON', '1W', '2W', '3W', '1M', '2M', '3M', '4M', '6M', '9M', '1Y']
    fx_options_trading_tenor = '1M'
    fx_options_cum_index = 'mult'
    fx_options_index_premium_output = 'pct-for'
    fx_options_index_strike = 'atm'
    fx_options_index_contract_type = 'european-call'
    fx_options_freeze_implied_vol = False
    fx_options_roll_event = 'expiry-date'
    fx_options_roll_days_before = 5
    fx_options_roll_months = 1
    fx_options_vol_function_type = 'CLARK5'
    fx_options_depo_tenor = '1M'
    fx_options_atm_method = 'fwd-delta-neutral-premium-adj'
    fx_options_delta_method = 'spot-delta-prem-adj'
    fx_options_alpha = 0.5
    fx_options_premium_output = 'pct-for'
    fx_options_delta_output = 'pct-fwd-delta-prem-adj'
    fx_options_solver = 'nelmer-mead-numba'
    fx_options_pricing_engine = 'financepy'
    fx_options_tol = 1e-08
    override_fields = {}

    def __init__(self, override_fields={}):
        if False:
            i = 10
            return i + 15
        try:
            from finmarketpy.util.marketcred import MarketCred
            cred_keys = MarketCred.__dict__.keys()
            for k in MarketConstants.__dict__.keys():
                if k in cred_keys and '__' not in k:
                    setattr(MarketConstants, k, getattr(MarketCred, k))
        except:
            pass
        if override_fields == {}:
            override_fields = MarketConstants.override_fields
        else:
            MarketConstants.override_fields = override_fields
        for k in override_fields.keys():
            if '__' not in k:
                setattr(MarketConstants, k, override_fields[k])