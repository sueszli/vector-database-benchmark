import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils
from random import randint
from random import uniform
from random import shuffle

def gen_data():
    if False:
        print('Hello World!')
    floatA = []
    intA = []
    sizeMat = range(0, 64)
    numZeros = 500
    numNans = 0
    numInfs = 500
    if numNans > 0:
        floatA = [float('NaN')] * numNans
        intA = [float('NaN')] * numNans
    if numInfs > 0:
        floatA.extend([float('inf')] * numInfs)
        intA.extend([float('inf')] * numInfs)
        floatA.extend([-1.0 * float('inf')] * numInfs)
        intA.extend([-1 * float('inf')] * numInfs)
    for index in range(numZeros):
        floatA.append(0.0 * randint(-1, 1))
        intA.append(0 * randint(-1, 1))
    for rad in sizeMat:
        tempInt = pow(2, rad)
        tempIntN = pow(2, rad + 1)
        intA.append(tempInt)
        intA.append(-1 * tempInt)
        randInt = randint(tempInt, tempIntN)
        intA.append(randInt)
        intA.append(-1 * randInt)
        intA.append(randint(tempInt, tempIntN))
        intA.append(-1 * randint(tempInt, tempIntN))
        floatA.append(tempInt * 1.0)
        floatA.append(-1.0 * tempInt)
        tempD = uniform(tempInt, tempIntN)
        floatA.append(tempD)
        floatA.append(-1.0 * tempD)
        floatA.append(uniform(tempInt, tempIntN))
        floatA.append(-1.0 * uniform(tempInt, tempIntN))
    intA.extend(intA)
    intA.extend(intA)
    intA.extend(intA)
    shuffle(intA)
    floatA.extend(floatA)
    floatA.extend(floatA)
    floatA.extend(floatA)
    shuffle(floatA)
    intFrame = h2o.H2OFrame(python_obj=intA)
    floatFrame = h2o.H2OFrame(python_obj=floatA)
    finalFrame = intFrame.concat([intFrame, floatFrame])
if __name__ == '__main__':
    pyunit_utils.standalone_test(gen_data)
else:
    gen_data()