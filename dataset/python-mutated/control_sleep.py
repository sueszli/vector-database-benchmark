from calibre.constants import ismacos, iswindows
if iswindows:
    from calibre_extensions.winutil import ES_CONTINUOUS, ES_DISPLAY_REQUIRED, ES_SYSTEM_REQUIRED, set_thread_execution_state

    def prevent_sleep(reason=''):
        if False:
            for i in range(10):
                print('nop')
        set_thread_execution_state(ES_CONTINUOUS | ES_DISPLAY_REQUIRED | ES_SYSTEM_REQUIRED)
        return 1

    def allow_sleep(cookie):
        if False:
            i = 10
            return i + 15
        set_thread_execution_state(ES_CONTINUOUS)
elif ismacos:
    from calibre_extensions.cocoa import create_io_pm_assertion, kIOPMAssertionTypeNoDisplaySleep, release_io_pm_assertion

    def prevent_sleep(reason=''):
        if False:
            print('Hello World!')
        return create_io_pm_assertion(kIOPMAssertionTypeNoDisplaySleep, reason or 'E-book viewer automated reading in progress')

    def allow_sleep(cookie):
        if False:
            print('Hello World!')
        release_io_pm_assertion(cookie)
else:

    def prevent_sleep(reason=''):
        if False:
            for i in range(10):
                print('nop')
        return 0

    def allow_sleep(cookie):
        if False:
            i = 10
            return i + 15
        pass