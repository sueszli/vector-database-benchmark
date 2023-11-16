import torch
from .distiller_utils import *
import logging
logging.captureWarnings(True)

def model_find_param_name(model, param_to_find):
    if False:
        i = 10
        return i + 15
    'Look up the name of a model parameter.\n\n    Arguments:\n        model: the model to search\n        param_to_find: the parameter whose name we want to look up\n\n    Returns:\n        The parameter name (string) or None, if the parameter was not found.\n    '
    for (name, param) in model.named_parameters():
        if param is param_to_find:
            return name
    return None

def model_find_module_name(model, module_to_find):
    if False:
        for i in range(10):
            print('nop')
    'Look up the name of a module in a model.\n\n    Arguments:\n        model: the model to search\n        module_to_find: the module whose name we want to look up\n\n    Returns:\n        The module name (string) or None, if the module was not found.\n    '
    for (name, m) in model.named_modules():
        if m == module_to_find:
            return name
    return None

def model_find_param(model, param_to_find_name):
    if False:
        print('Hello World!')
    'Look a model parameter by its name\n\n    Arguments:\n        model: the model to search\n        param_to_find_name: the name of the parameter that we are searching for\n\n    Returns:\n        The parameter or None, if the paramter name was not found.\n    '
    for (name, param) in model.named_parameters():
        if name == param_to_find_name:
            return param
    return None

def model_find_module(model, module_to_find):
    if False:
        while True:
            i = 10
    'Given a module name, find the module in the provided model.\n\n    Arguments:\n        model: the model to search\n        module_to_find: the module whose name we want to look up\n\n    Returns:\n        The module or None, if the module was not found.\n    '
    for (name, m) in model.named_modules():
        if name == module_to_find:
            return m
    return None