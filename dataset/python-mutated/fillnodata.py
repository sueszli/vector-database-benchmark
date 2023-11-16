import numpy as np
DTYPEf = np.float32

def replace_nans(array, max_iter, tolerance, kernel_size=1, method='localmean'):
    if False:
        i = 10
        return i + 15
    '\n\tReplace NaN elements in an array using an iterative image inpainting algorithm.\n\tThe algorithm is the following:\n\t1) For each element in the input array, replace it by a weighted average\n\tof the neighbouring elements which are not NaN themselves. The weights depends\n\tof the method type. If ``method=localmean`` weight are equal to 1/( (2*kernel_size+1)**2 -1 )\n\t2) Several iterations are needed if there are adjacent NaN elements.\n\tIf this is the case, information is "spread" from the edges of the missing\n\tregions iteratively, until the variation is below a certain threshold.\n\n\tParameters\n\t----------\n\tarray : 2d np.ndarray\n\tan array containing NaN elements that have to be replaced\n\n\tmax_iter : int\n\tthe number of iterations\n\n\tkernel_size : int\n\tthe size of the kernel, default is 1\n\n\tmethod : str\n\tthe method used to replace invalid values. Valid options are \'localmean\', \'idw\'.\n\n\tReturns\n\t-------\n\tfilled : 2d np.ndarray\n\ta copy of the input array, where NaN elements have been replaced.\n\t'
    filled = np.empty([array.shape[0], array.shape[1]], dtype=DTYPEf)
    kernel = np.empty((2 * kernel_size + 1, 2 * kernel_size + 1), dtype=DTYPEf)
    (inans, jnans) = np.nonzero(np.isnan(array))
    n_nans = len(inans)
    replaced_new = np.zeros(n_nans, dtype=DTYPEf)
    replaced_old = np.zeros(n_nans, dtype=DTYPEf)
    if method == 'localmean':
        for i in range(2 * kernel_size + 1):
            for j in range(2 * kernel_size + 1):
                kernel[i, j] = 1
    elif method == 'idw':
        kernel = np.array([[0, 0.5, 0.5, 0.5, 0], [0.5, 0.75, 0.75, 0.75, 0.5], [0.5, 0.75, 1, 0.75, 0.5], [0.5, 0.75, 0.75, 0.5, 1], [0, 0.5, 0.5, 0.5, 0]])
    else:
        raise ValueError("method not valid. Should be one of 'localmean', 'idw'.")
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            filled[i, j] = array[i, j]
    for it in range(max_iter):
        for k in range(n_nans):
            i = inans[k]
            j = jnans[k]
            filled[i, j] = 0.0
            n = 0
            for I in range(2 * kernel_size + 1):
                for J in range(2 * kernel_size + 1):
                    if i + I - kernel_size < array.shape[0] and i + I - kernel_size >= 0:
                        if j + J - kernel_size < array.shape[1] and j + J - kernel_size >= 0:
                            if filled[i + I - kernel_size, j + J - kernel_size] == filled[i + I - kernel_size, j + J - kernel_size]:
                                if I - kernel_size != 0 and J - kernel_size != 0:
                                    filled[i, j] = filled[i, j] + filled[i + I - kernel_size, j + J - kernel_size] * kernel[I, J]
                                    n = n + 1 * kernel[I, J]
            if n != 0:
                filled[i, j] = filled[i, j] / n
                replaced_new[k] = filled[i, j]
            else:
                filled[i, j] = np.nan
        if np.mean((replaced_new - replaced_old) ** 2) < tolerance:
            break
        else:
            for l in range(n_nans):
                replaced_old[l] = replaced_new[l]
    return filled

def sincinterp(image, x, y, kernel_size=3):
    if False:
        for i in range(10):
            print('nop')
    '\n\tRe-sample an image at intermediate positions between pixels.\n\tThis function uses a cardinal interpolation formula which limits\n\tthe loss of information in the resampling process. It uses a limited\n\tnumber of neighbouring pixels.\n\n\tThe new image :math:`im^+` at fractional locations :math:`x` and :math:`y` is computed as:\n\t.. math::\n\tim^+(x,y) = \\sum_{i=-\\mathtt{kernel\\_size}}^{i=\\mathtt{kernel\\_size}} \\sum_{j=-\\mathtt{kernel\\_size}}^{j=\\mathtt{kernel\\_size}} \\mathtt{image}(i,j) sin[\\pi(i-\\mathtt{x})] sin[\\pi(j-\\mathtt{y})] / \\pi(i-\\mathtt{x}) / \\pi(j-\\mathtt{y})\n\n\tParameters\n\t----------\n\timage : np.ndarray, dtype np.int32\n\tthe image array.\n\n\tx : two dimensions np.ndarray of floats\n\tan array containing fractional pixel row\n\tpositions at which to interpolate the image\n\n\ty : two dimensions np.ndarray of floats\n\tan array containing fractional pixel column\n\tpositions at which to interpolate the image\n\n\tkernel_size : int\n\tinterpolation is performed over a ``(2*kernel_size+1)*(2*kernel_size+1)``\n\tsubmatrix in the neighbourhood of each interpolation point.\n\n\tReturns\n\t-------\n\tim : np.ndarray, dtype np.float64\n\tthe interpolated value of ``image`` at the points specified by ``x`` and ``y``\n\t'
    r = np.zeros([x.shape[0], x.shape[1]], dtype=DTYPEf)
    pi = 3.1419
    for I in range(x.shape[0]):
        for J in range(x.shape[1]):
            for i in range(int(x[I, J]) - kernel_size, int(x[I, J]) + kernel_size + 1):
                for j in range(int(y[I, J]) - kernel_size, int(y[I, J]) + kernel_size + 1):
                    if i >= 0 and i <= image.shape[0] and (j >= 0) and (j <= image.shape[1]):
                        if i - x[I, J] == 0.0 and j - y[I, J] == 0.0:
                            r[I, J] = r[I, J] + image[i, j]
                        elif i - x[I, J] == 0.0:
                            r[I, J] = r[I, J] + image[i, j] * np.sin(pi * (j - y[I, J])) / (pi * (j - y[I, J]))
                        elif j - y[I, J] == 0.0:
                            r[I, J] = r[I, J] + image[i, j] * np.sin(pi * (i - x[I, J])) / (pi * (i - x[I, J]))
                        else:
                            r[I, J] = r[I, J] + image[i, j] * np.sin(pi * (i - x[I, J])) * np.sin(pi * (j - y[I, J])) / (pi * pi * (i - x[I, J]) * (j - y[I, J]))
    return r