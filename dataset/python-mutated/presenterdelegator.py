from sentry.runner.commands.presenters.consolepresenter import ConsolePresenter
from sentry.runner.commands.presenters.slackpresenter import SlackPresenter

class PresenterDelegator:

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self._consolepresenter = ConsolePresenter()
        self._slackpresenter = None
        if SlackPresenter.is_slack_enabled():
            self._slackpresenter = SlackPresenter()

    def __getattr__(self, attr: str):
        if False:
            i = 10
            return i + 15

        def wrapper(*args, **kwargs):
            if False:
                return 10
            getattr(self._consolepresenter, attr)(*args, **kwargs)
            if self._slackpresenter:
                getattr(self._slackpresenter, attr)(*args, **kwargs)
        return wrapper