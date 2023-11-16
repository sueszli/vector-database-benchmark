import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils
import random

def get_set_list_timezones():
    if False:
        for i in range(10):
            print('nop')
    origTZ = h2o.get_timezone()
    print('Original timezone: {0}'.format(origTZ))
    timezones = h2o.list_timezones()
    print('timezones[0]:', timezones[0])
    zone = timezones[random.randint(1, timezones.nrow - 1), 0].split(' ')[1].split(',')[0]
    print('Setting the timezone: {0}'.format(zone))
    h2o.set_timezone(zone)
    newTZ = h2o.get_timezone()
    assert newTZ == zone, 'Expected new timezone to be {0}, but got {01}'.format(zone, newTZ)
    print('Setting the timezone back to original: {0}'.format(origTZ))
    h2o.set_timezone(origTZ)
if __name__ == '__main__':
    pyunit_utils.standalone_test(get_set_list_timezones)
else:
    get_set_list_timezones()