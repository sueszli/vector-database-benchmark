import numpy

def vibr_energy(harmonic, anharmonic, i):
    if False:
        return 10
    return numpy.exp(-harmonic * i - anharmonic * i ** 2)