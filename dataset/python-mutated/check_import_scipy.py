def check_import_scipy(OsName):
    if False:
        print('Hello World!')
    print_info = ''
    if OsName == 'nt':
        try:
            import scipy.io as scio
        except ImportError as e:
            print_info = str(e)
        if len(print_info) > 0:
            if 'DLL load failed' in print_info:
                raise ImportError(print_info + '\nplease download Visual C++ Redistributable from https://support.microsoft.com/en-us/topic/the-latest-supported-visual-c-downloads-2647da03-1eea-4433-9aff-95f26a218cc0')