try:
    from django.conf import settings  # noqa: F401

    from analytics.models import *  # noqa: F403
    from zerver.models import *  # noqa: F403
except Exception:
    import traceback

    print("\nException importing Zulip core modules on startup!")
    traceback.print_exc()
else:
    print("\nSuccessfully imported Zulip settings, models, and actions functions.")
