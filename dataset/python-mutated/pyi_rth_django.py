def _pyi_rthook():
    if False:
        return 10
    import django.utils.autoreload
    _old_restart_with_reloader = django.utils.autoreload.restart_with_reloader

    def _restart_with_reloader(*args):
        if False:
            i = 10
            return i + 15
        import sys
        a0 = sys.argv.pop(0)
        try:
            return _old_restart_with_reloader(*args)
        finally:
            sys.argv.insert(0, a0)
    django.utils.autoreload.restart_with_reloader = _restart_with_reloader
_pyi_rthook()
del _pyi_rthook