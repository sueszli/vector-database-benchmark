import dateutil
import tzlocal

class TimezoneError(Exception):
    pass

def mktz(zone=None):
    if False:
        return 10
    "\n    Return a new timezone (tzinfo object) based on the zone using the python-dateutil\n    package.\n\n    The concise name 'mktz' is for convenient when using it on the\n    console.\n\n    Parameters\n    ----------\n    zone : `String`\n           The zone for the timezone. This defaults to local, returning:\n           tzlocal.get_localzone()\n\n    Returns\n    -------\n    An instance of a timezone which implements the tzinfo interface.\n\n    Raises\n    - - - - - -\n    TimezoneError : Raised if a user inputs a bad timezone name.\n    "
    if zone is None:
        zone = tzlocal.get_localzone().zone
    tz = dateutil.tz.gettz(zone)
    if not tz:
        raise TimezoneError('Timezone "%s" can not be read' % zone)
    if not hasattr(tz, 'zone'):
        tz.zone = zone
        for p in dateutil.tz.TZPATHS:
            if zone.startswith(p):
                tz.zone = zone[len(p) + 1:]
                break
    return tz