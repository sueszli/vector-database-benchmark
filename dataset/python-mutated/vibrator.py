import jnius
__all__ = ['vibrate']

def vibrate(pattern, repeat=None):
    if False:
        print('Hello World!')
    ' take a list of int as pattern '
    PythonActivity = jnius.autoclass('org.renpy.android.PythonService')
    Context = jnius.autoclass('android.content.Context')
    activity = PythonActivity.mService
    vibrator = activity.getSystemService(Context.VIBRATOR_SERVICE)
    if vibrator.hasVibrator():
        try:
            if repeat:
                vibrator.vibrate(list(pattern), repeat)
            else:
                vibrator.vibrate(list(pattern), -1)
        except KeyboardInterrupt:
            vibrator.cancel()
            raise
    else:
        raise RuntimeError('The device does not have a vibrator')