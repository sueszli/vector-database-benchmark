from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import string

def import_in_function():
    if False:
        i = 10
        return i + 15
    import symtable
    import os, sys
    import time as thyme
    import random as rand, socket as sock
    from collections import defaultdict
    from math import sin as sign, cos as cosplay

class ClassWithImports:
    import tokenize

    def __init__(self):
        if False:
            return 10
        import trace