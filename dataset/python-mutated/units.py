"""``geopy.units`` module provides utility functions for performing
angle and distance unit conversions.

Some shortly named aliases are provided for convenience (e.g.
:func:`.km` is an alias for :func:`.kilometers`).
"""
import math

def degrees(radians=0, arcminutes=0, arcseconds=0):
    if False:
        return 10
    '\n    Convert angle to degrees.\n    '
    deg = 0.0
    if radians:
        deg = math.degrees(radians)
    if arcminutes:
        deg += arcminutes / arcmin(degrees=1.0)
    if arcseconds:
        deg += arcseconds / arcsec(degrees=1.0)
    return deg

def radians(degrees=0, arcminutes=0, arcseconds=0):
    if False:
        while True:
            i = 10
    '\n    Convert angle to radians.\n    '
    if arcminutes:
        degrees += arcminutes / arcmin(degrees=1.0)
    if arcseconds:
        degrees += arcseconds / arcsec(degrees=1.0)
    return math.radians(degrees)

def arcminutes(degrees=0, radians=0, arcseconds=0):
    if False:
        while True:
            i = 10
    '\n    Convert angle to arcminutes.\n    '
    if radians:
        degrees += math.degrees(radians)
    if arcseconds:
        degrees += arcseconds / arcsec(degrees=1.0)
    return degrees * 60.0

def arcseconds(degrees=0, radians=0, arcminutes=0):
    if False:
        i = 10
        return i + 15
    '\n    Convert angle to arcseconds.\n    '
    if radians:
        degrees += math.degrees(radians)
    if arcminutes:
        degrees += arcminutes / arcmin(degrees=1.0)
    return degrees * 3600.0

def kilometers(meters=0, miles=0, feet=0, nautical=0):
    if False:
        i = 10
        return i + 15
    '\n    Convert distance to kilometers.\n    '
    ret = 0.0
    if meters:
        ret += meters / 1000.0
    if feet:
        ret += feet / ft(1.0)
    if nautical:
        ret += nautical / nm(1.0)
    ret += miles * 1.609344
    return ret

def meters(kilometers=0, miles=0, feet=0, nautical=0):
    if False:
        print('Hello World!')
    '\n    Convert distance to meters.\n    '
    return (kilometers + km(nautical=nautical, miles=miles, feet=feet)) * 1000

def miles(kilometers=0, meters=0, feet=0, nautical=0):
    if False:
        return 10
    '\n    Convert distance to miles.\n    '
    ret = 0.0
    if nautical:
        kilometers += nautical / nm(1.0)
    if feet:
        kilometers += feet / ft(1.0)
    if meters:
        kilometers += meters / 1000.0
    ret += kilometers / 1.609344
    return ret

def feet(kilometers=0, meters=0, miles=0, nautical=0):
    if False:
        i = 10
        return i + 15
    '\n    Convert distance to feet.\n    '
    ret = 0.0
    if nautical:
        kilometers += nautical / nm(1.0)
    if meters:
        kilometers += meters / 1000.0
    if kilometers:
        miles += mi(kilometers=kilometers)
    ret += miles * 5280
    return ret

def nautical(kilometers=0, meters=0, miles=0, feet=0):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert distance to nautical miles.\n    '
    ret = 0.0
    if feet:
        kilometers += feet / ft(1.0)
    if miles:
        kilometers += km(miles=miles)
    if meters:
        kilometers += meters / 1000.0
    ret += kilometers / 1.852
    return ret
rad = radians
arcmin = arcminutes
arcsec = arcseconds
km = kilometers
m = meters
mi = miles
ft = feet
nm = nautical