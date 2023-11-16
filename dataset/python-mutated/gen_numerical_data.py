import sys
import h2o
from random import randint
from random import uniform
from random import shuffle
import numpy as np

def gen_data():
    if False:
        return 10
    floatA = []
    intA = []
    sizeMat = range(0, 30)
    lowBoundF = -100000
    upperBoundF = -1 * lowBoundF
    upperBoundL = pow(2, 35)
    lowBoundL = upperBoundL - 100000
    numZeros = 0
    numNans = 0
    numInfs = 500
    numRep = 2
    csvFile = '/Users/wendycwong/temp/TopBottomNRep4.csv'
    fMult = 1.1
    fintA = []
    ffloatA = []
    for ind in range(0, 1000):
        floatA = []
        intA = []
        genRandomData(intA, floatA, sizeMat)
        fintA.extend(intA)
        ffloatA.extend(floatA)
    shuffle(fintA)
    shuffle(ffloatA)
    bottom20FrameL = h2o.H2OFrame(python_obj=zip(fintA))
    bottom20FrameF = h2o.H2OFrame(python_obj=zip(ffloatA))
    h2o.download_csv(bottom20FrameL.cbind(bottom20FrameF), '/Users/wendycwong/temp/smallIntFloats.csv')
    genStaticData(intA, floatA, upperBoundL, lowBoundF, upperBoundF, fMult)
    tempL = intA[0:int(round(len(intA) * 0.2))]
    tempF = floatA[0:int(round(len(floatA) * 0.2))]
    bottom20FrameL = h2o.H2OFrame(python_obj=zip(tempL))
    bottom20FrameF = h2o.H2OFrame(python_obj=zip(tempF))
    h2o.download_csv(bottom20FrameL.cbind(bottom20FrameF), '/Users/wendycwong/temp/Bottom20Per.csv')
    tempL = intA[int(round(len(intA) * 0.8)):len(intA)]
    tempL.sort()
    tempF = floatA[int(round(len(floatA) * 0.8)):len(floatA)]
    tempF.sort()
    bottom20FrameL = h2o.H2OFrame(python_obj=zip(tempL))
    bottom20FrameF = h2o.H2OFrame(python_obj=zip(tempF))
    h2o.download_csv(bottom20FrameL.cbind(bottom20FrameF), '/Users/wendycwong/temp/Top20Per.csv')
    for val in range(0, numRep):
        intA.extend(intA)
        floatA.extend(floatA)
    shuffle(intA)
    shuffle(floatA)
    intFrame = h2o.H2OFrame(python_obj=zip(intA))
    floatFrame = h2o.H2OFrame(python_obj=zip(floatA))
    h2o.download_csv(intFrame.cbind(floatFrame), csvFile)

def genDataFrame(sizeMat, lowBound, uppderBound, numRep, numZeros, numNans, numInfs):
    if False:
        while True:
            i = 10
    '\n    This function will generate an H2OFrame of two columns.  One column will be float and the other will\n    be long.\n    \n    :param sizeMat: integer denoting size of bounds\n    :param lowBound: lower bound\n    :param uppderBound: \n    :param trueRandom: \n    :param numRep: number of times to repeat arrays in order to generate duplicated rows\n    :param numZeros: \n    :param numNans: \n    :param numInfs: \n    :return: \n    '
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

def genRandomData(intA, floatA, sizeMat):
    if False:
        print('Hello World!')
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

def genStaticData(intA, floatA, upperBoundL, lowBoundF, upperBoundF, fMult):
    if False:
        return 10
    for val in range(lowBoundF, upperBoundF):
        floatA.append(val * fMult)
        intA.append(upperBoundL)
        upperBoundL = upperBoundL - 1
    intA.reverse()

def genMergedSeparaData(MergedRows, intUpper, intLow, doubleUpper, doubleLow, bProb):
    if False:
        print('Hello World!')
    merged = h2o.create_frame(rows=MergedRows, cols=3, integer_fraction=1, integer_range=intUpper - intLow)
    print('Done, save with Flow')

def main(argv):
    if False:
        print('Hello World!')
    h2o.init(strict_version_check=False)
    genMergedSeparaData(2000000000, pow(2, 30), -1 * pow(2, 30), 1.0 * pow(2, 63), -1.0 * pow(2, 63), 0.8)
if __name__ == '__main__':
    main(sys.argv)