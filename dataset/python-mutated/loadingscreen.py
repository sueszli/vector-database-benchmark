from jnius import autoclass
from android.config import ACTIVITY_CLASS_NAME

def hide_loading_screen():
    if False:
        return 10
    mActivity = autoclass(ACTIVITY_CLASS_NAME).mActivity
    mActivity.removeLoadingScreen()