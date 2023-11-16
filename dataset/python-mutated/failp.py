from errbot import BotPlugin, ValidationException

class FailP(BotPlugin):
    """
    Just a plugin failing at config time.
    """

    def get_configuration_template(self):
        if False:
            return 10
        return {'One': 1, 'Two': 2}

    def check_configuration(self, configuration):
        if False:
            i = 10
            return i + 15
        raise ValidationException('Message explaining why it failed.')