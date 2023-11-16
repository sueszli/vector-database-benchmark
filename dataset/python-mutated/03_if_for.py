def get_config_vars(_config_vars, args):
    if False:
        for i in range(10):
            print('nop')
    if _config_vars:
        if args == 1:
            if args < 8:
                for key in ('LDFLAGS', 'BASECFLAGS'):
                    _config_vars[key] = 4