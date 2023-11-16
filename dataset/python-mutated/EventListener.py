class EventListener:
    """
    Base event listener class
    """

    def on_event(self, event, extension):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param ~ulauncher.api.shared.event.BaseEvent event: event that listener was subscribed to\n        :param ~ulauncher.api.Extension extension:\n\n        :rtype: bool, strict, dict or None\n        :return: Action to run\n        '