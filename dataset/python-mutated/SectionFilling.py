import copy
from inspect import signature
import logging
from coalib.bears.BEAR_KIND import BEAR_KIND
from coalib.collecting import Dependencies
from coalib.collecting.Collectors import collect_bears, collect_bears_by_aspects
from coalib.settings.Setting import Setting

def fill_section(section, acquire_settings, log_printer, bears):
    if False:
        while True:
            i = 10
    '\n    Retrieves needed settings from given bears and asks the user for\n    missing values.\n\n    If a setting is requested by several bears, the help text from the\n    latest bear will be taken.\n\n    :param section:          A section containing available settings. Settings\n                             will be added if some are missing.\n    :param acquire_settings: The method to use for requesting settings. It will\n                             get a parameter which is a dictionary with the\n                             settings name as key and a list containing a\n                             description in [0] and the names of the bears\n                             who need this setting in all following indexes.\n    :param log_printer:      The log printer for logging.\n    :param bears:            All bear classes or instances.\n    :return:                 The new section.\n    '
    prel_needed_settings = {}
    for bear in bears:
        needed = bear.get_non_optional_settings()
        for key in needed:
            if key in prel_needed_settings:
                prel_needed_settings[key].append(bear.name)
            else:
                prel_needed_settings[key] = [needed[key][0], bear.name]
    needed_settings = {}
    for (setting, help_text) in prel_needed_settings.items():
        if setting not in section:
            needed_settings[setting] = help_text
    if len(needed_settings) > 0:
        if len(signature(acquire_settings).parameters) == 2:
            new_vals = acquire_settings(None, needed_settings)
        else:
            logging.warning('acquire_settings: section parameter is deprecated.')
            new_vals = acquire_settings(None, needed_settings, section)
        for (setting, help_text) in new_vals.items():
            section.append(Setting(setting, help_text))
    return section

def fill_settings(sections, acquire_settings, log_printer=None, fill_section_method=fill_section, targets=None, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Retrieves all bears and requests missing settings via the given\n    acquire_settings method.\n\n    This will retrieve all bears and their dependencies.\n\n    :param sections:            The sections to fill up, modified in place.\n    :param acquire_settings:    The method to use for requesting settings. It\n                                will get a parameter which is a dictionary with\n                                the settings name as key and a list containing\n                                a description in [0] and the names of the bears\n                                who need this setting in all following indexes.\n    :param log_printer:         The log printer to use for logging.\n    :param fill_section_method: Method to be used to fill the section settings.\n    :param targets:             List of section names to be executed which are\n                                passed from cli.\n    :param kwargs:              Any other arguments for the fill_section_method\n                                can be supplied via kwargs, which are passed\n                                directly to the fill_section_method.\n    :return:                    A tuple containing (local_bears, global_bears),\n                                each of them being a dictionary with the\n                                section name as key and as value the bears as a\n                                list.\n    '
    local_bears = {}
    global_bears = {}
    for (section_name, section) in sections.items():
        bear_dirs = section.bear_dirs()
        if getattr(section, 'aspects', None):
            (section_local_bears, section_global_bears) = collect_bears_by_aspects(section.aspects, [BEAR_KIND.LOCAL, BEAR_KIND.GLOBAL])
        else:
            bears = list(section.get('bears', ''))
            (section_local_bears, section_global_bears) = collect_bears(bear_dirs, bears, [BEAR_KIND.LOCAL, BEAR_KIND.GLOBAL])
        section_local_bears = Dependencies.resolve(section_local_bears)
        section_global_bears = Dependencies.resolve(section_global_bears)
        all_bears = copy.deepcopy(section_local_bears)
        all_bears.extend(section_global_bears)
        if targets is None or section.is_enabled(targets):
            fill_section_method(section, acquire_settings, None, all_bears, **kwargs)
        local_bears[section_name] = section_local_bears
        global_bears[section_name] = section_global_bears
    return (local_bears, global_bears)