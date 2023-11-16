import numpy

def log_likelihood(data, mean, sigma):
    if False:
        i = 10
        return i + 15
    s = (data - mean) ** 2 / (2 * sigma ** 2)
    pdfs = numpy.exp(-s)
    pdfs /= numpy.sqrt(2 * numpy.pi) * sigma
    return numpy.log(pdfs).sum()