import logging

def fail_acquire_settings(log_printer, settings_names_dict):
    if False:
        i = 10
        return i + 15
    '\n    This method throws an exception if any setting needs to be acquired.\n\n    :param log_printer:         Printer responsible for logging the messages.\n    :param settings_names_dict: A dictionary with the settings name as key and\n                                a list containing a description in [0] and the\n                                name of the bears who need this setting in [1]\n                                and following.\n    :raises AssertionError:     If any setting is required.\n    :raises TypeError:          If ``settings_names_dict`` is not a\n                                dictionary.\n    '
    if not isinstance(settings_names_dict, dict):
        raise TypeError('The settings_names_dict parameter has to be a dictionary.')
    required_settings = settings_names_dict.keys()
    if len(required_settings) != 0:
        msg = 'During execution, we found that some required settings were not provided. They are:\n'
        for (name, setting) in settings_names_dict.items():
            msg += f'{name} (from {setting[1]}) - {setting[0]}'
        logging.error(msg)
        raise AssertionError(msg)