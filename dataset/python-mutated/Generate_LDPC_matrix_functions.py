import string
import sys
import numpy as np
from numpy.linalg import inv, det
from numpy.random import shuffle, randint
from numpy import zeros, array, linalg, vstack, dot, concatenate, identity, zeros_like, eye

def read_alist_file(filename):
    if False:
        while True:
            i = 10
    '\n    This function reads in an alist file and creates the\n    corresponding parity check matrix H. The format of alist\n    files is described at:\n    http://www.inference.phy.cam.ac.uk/mackay/codes/alist.html\n    '
    with open(filename, 'r') as myfile:
        data = myfile.readlines()
        (numCols, numRows) = parse_alist_header(data[0])
        H = zeros((numRows, numCols))
        for lineNumber in np.arange(4, 4 + numCols):
            indices = data[lineNumber].split()
            for index in indices:
                H[int(index) - 1, lineNumber - 4] = 1
        return H

def parse_alist_header(header):
    if False:
        return 10
    size = header.split()
    return (int(size[0]), int(size[1]))

def write_alist_file(filename, H, verbose=0):
    if False:
        while True:
            i = 10
    '\n    This function writes an alist file for the parity check\n    matrix. The format of alist files is described at:\n    http://www.inference.phy.cam.ac.uk/mackay/codes/alist.html\n    '
    with open(filename, 'w') as myfile:
        numRows = H.shape[0]
        numCols = H.shape[1]
        tempstring = repr(numCols) + ' ' + repr(numRows) + '\n'
        myfile.write(tempstring)
        tempstring1 = ''
        tempstring2 = ''
        maxRowWeight = 0
        for rowNum in np.arange(numRows):
            nonzeros = array(H[rowNum, :].nonzero())
            rowWeight = nonzeros.shape[1]
            if rowWeight > maxRowWeight:
                maxRowWeight = rowWeight
            tempstring1 = tempstring1 + repr(rowWeight) + ' '
            for tempArray in nonzeros:
                for index in tempArray:
                    tempstring2 = tempstring2 + repr(index + 1) + ' '
            tempstring2 = tempstring2 + '\n'
        tempstring1 = tempstring1 + '\n'
        tempstring3 = ''
        tempstring4 = ''
        maxColWeight = 0
        for colNum in np.arange(numCols):
            nonzeros = array(H[:, colNum].nonzero())
            colWeight = nonzeros.shape[1]
            if colWeight > maxColWeight:
                maxColWeight = colWeight
            tempstring3 = tempstring3 + repr(colWeight) + ' '
            for tempArray in nonzeros:
                for index in tempArray:
                    tempstring4 = tempstring4 + repr(index + 1) + ' '
            tempstring4 = tempstring4 + '\n'
        tempstring3 = tempstring3 + '\n'
        tempstring = repr(maxColWeight) + ' ' + repr(maxRowWeight) + '\n'
        myfile.write(tempstring)
        myfile.write(tempstring3)
        myfile.write(tempstring1)
        myfile.write(tempstring4)
        myfile.write(tempstring2)

class LDPC_matrix(object):
    """ Class for a LDPC parity check matrix """

    def __init__(self, alist_filename=None, n_p_q=None, H_matrix=None):
        if False:
            for i in range(10):
                print('nop')
        if alist_filename != None:
            self.H = read_alist_file(alist_filename)
        elif n_p_q != None:
            self.H = self.regular_LDPC_code_contructor(n_p_q)
        elif H_matrix != None:
            self.H = H_matrix
        else:
            print('Error: provide either an alist filename, ', end='')
            print('parameters for constructing regular LDPC parity, ', end='')
            print('check matrix, or a numpy array.')
        self.rank = linalg.matrix_rank(self.H)
        self.numRows = self.H.shape[0]
        self.n = self.H.shape[1]
        self.k = self.n - self.numRows

    def regular_LDPC_code_contructor(self, n_p_q):
        if False:
            for i in range(10):
                print('nop')
        "\n        This function constructs a LDPC parity check matrix\n        H. The algorithm follows Gallager's approach where we create\n        p submatrices and stack them together. Reference: Turbo\n        Coding for Satellite and Wireless Communications, section\n        9,3.\n\n        Note: the matrices computed from this algorithm will never\n        have full rank. (Reference Gallager's Dissertation.) They\n        will have rank = (number of rows - p + 1). To convert it\n        to full rank, use the function get_full_rank_H_matrix\n        "
        n = n_p_q[0]
        p = n_p_q[1]
        q = n_p_q[2]
        ratioTest = n * 1.0 / q
        if ratioTest % 1 != 0:
            print('\nError in regular_LDPC_code_contructor: The ', end='')
            print('ratio of inputs n/q must be a whole number.\n')
            return
        m = n * p // q
        submatrix1 = zeros((m // p, n))
        for row in np.arange(m // p):
            range1 = row * q
            range2 = (row + 1) * q
            submatrix1[row, range1:range2] = 1
            H = submatrix1
        submatrixNum = 2
        newColumnOrder = np.arange(n)
        while submatrixNum <= p:
            submatrix = zeros((m // p, n))
            shuffle(newColumnOrder)
            for columnNum in np.arange(n):
                submatrix[:, columnNum] = submatrix1[:, newColumnOrder[columnNum]]
            H = vstack((H, submatrix))
            submatrixNum = submatrixNum + 1
        size = H.shape
        rows = size[0]
        cols = size[1]
        for rowNum in np.arange(rows):
            nonzeros = array(H[rowNum, :].nonzero())
            if nonzeros.shape[1] != q:
                print('Row', rowNum, 'has incorrect weight!')
                return
        for columnNum in np.arange(cols):
            nonzeros = array(H[:, columnNum].nonzero())
            if nonzeros.shape[1] != p:
                print('Row', columnNum, 'has incorrect weight!')
                return
        return H

def greedy_upper_triangulation(H, verbose=0):
    if False:
        for i in range(10):
            print('nop')
    '\n    This function performs row/column permutations to bring\n    H into approximate upper triangular form via greedy\n    upper triangulation method outlined in Modern Coding\n    Theory Appendix 1, Section A.2\n    '
    H_t = H.copy()
    if linalg.matrix_rank(H_t) != H_t.shape[0]:
        print('Rank of H:', linalg.matrix_rank(tempArray))
        print('H has', H_t.shape[0], 'rows')
        print('Error: H must be full rank.')
        return
    size = H_t.shape
    n = size[1]
    k = n - size[0]
    g = t = 0
    while t != n - k - g:
        H_residual = H_t[t:n - k - g, t:n]
        size = H_residual.shape
        numRows = size[0]
        numCols = size[1]
        minResidualDegrees = zeros((1, numCols), dtype=int)
        for colNum in np.arange(numCols):
            nonZeroElements = array(H_residual[:, colNum].nonzero())
            minResidualDegrees[0, colNum] = nonZeroElements.shape[1]
        nonZeroElementIndices = minResidualDegrees.nonzero()
        nonZeroElements = minResidualDegrees[nonZeroElementIndices[0], nonZeroElementIndices[1]]
        minimumResidualDegree = nonZeroElements.min()
        indices = (minResidualDegrees == minimumResidualDegree).nonzero()[1]
        indices = indices + t
        if indices.shape[0] == 1:
            columnC = indices[0]
        else:
            randomIndex = randint(0, indices.shape[0], (1, 1))[0][0]
            columnC = indices[randomIndex]
        Htemp = H_t.copy()
        if minimumResidualDegree == 1:
            rowThatContainsNonZero = H_residual[:, columnC - t].nonzero()[0][0]
            Htemp[:, columnC] = H_t[:, t]
            Htemp[:, t] = H_t[:, columnC]
            H_t = Htemp.copy()
            Htemp = H_t.copy()
            Htemp[rowThatContainsNonZero + t, :] = H_t[t, :]
            Htemp[t, :] = H_t[rowThatContainsNonZero + t, :]
            H_t = Htemp.copy()
            Htemp = H_t.copy()
        else:
            rowsThatContainNonZeros = H_residual[:, columnC - t].nonzero()[0]
            Htemp[:, columnC] = H_t[:, t]
            Htemp[:, t] = H_t[:, columnC]
            H_t = Htemp.copy()
            Htemp = H_t.copy()
            r1 = rowsThatContainNonZeros[0]
            Htemp[r1 + t, :] = H_t[t, :]
            Htemp[t, :] = H_t[r1 + t, :]
            numRowsLeft = rowsThatContainNonZeros.shape[0] - 1
            H_t = Htemp.copy()
            Htemp = H_t.copy()
            for index in np.arange(1, numRowsLeft + 1):
                rowInH_residual = rowsThatContainNonZeros[index]
                rowInH_t = rowInH_residual + t - index + 1
                m = n - k
                Htemp[m - 1, :] = H_t[rowInH_t, :]
                sub_index = 1
                while sub_index < m - rowInH_t:
                    Htemp[m - sub_index - 1, :] = H_t[m - sub_index, :]
                    sub_index = sub_index + 1
                H_t = Htemp.copy()
                Htemp = H_t.copy()
            H_t = Htemp.copy()
            Htemp = H_t.copy()
            g = g + (minimumResidualDegree - 1)
        t = t + 1
    if g == 0:
        if verbose:
            print('Error: gap is 0.')
        return
    T = H_t[0:t, 0:t]
    E = H_t[t:t + g, 0:t]
    A = H_t[0:t, t:t + g]
    C = H_t[t:t + g, t:t + g]
    D = H_t[t:t + g, t + g:n]
    invTmod2array = inv_mod2(T)
    temp1 = dot(E, invTmod2array) % 2
    temp2 = dot(temp1, A) % 2
    phi = (C - temp2) % 2
    if phi.any():
        try:
            invPhi = inv_mod2(phi)
        except linalg.linalg.LinAlgError:
            if verbose > 1:
                print('Initial phi is singular')
        else:
            if verbose > 1:
                print('Initial phi is nonsingular')
            return [H_t, g, t]
    elif verbose:
        print('Initial phi is all zeros:\n', phi)
    if not (C.any() or D.any()):
        if verbose:
            print('C and D are all zeros. There is no hope in')
            print('finding a nonsingular phi matrix. ')
        return
    maxIterations = 300
    iterationCount = 0
    columnsToShuffle = np.arange(t, n)
    rowsToShuffle = np.arange(t, t + g)
    while iterationCount < maxIterations:
        if verbose > 1:
            print('iterationCount:', iterationCount)
        tempH = H_t.copy()
        shuffle(columnsToShuffle)
        shuffle(rowsToShuffle)
        index = 0
        for newDestinationColumnNumber in np.arange(t, n):
            oldColumnNumber = columnsToShuffle[index]
            tempH[:, newDestinationColumnNumber] = H_t[:, oldColumnNumber]
            index += 1
        tempH2 = tempH.copy()
        index = 0
        for newDesinationRowNumber in np.arange(t, t + g):
            oldRowNumber = rowsToShuffle[index]
            tempH[newDesinationRowNumber, :] = tempH2[oldRowNumber, :]
            index += 1
        H_t = tempH.copy()
        T = H_t[0:t, 0:t]
        E = H_t[t:t + g, 0:t]
        A = H_t[0:t, t:t + g]
        C = H_t[t:t + g, t:t + g]
        invTmod2array = inv_mod2(T)
        temp1 = dot(E, invTmod2array) % 2
        temp2 = dot(temp1, A) % 2
        phi = (C - temp2) % 2
        if phi.any():
            try:
                invPhi = inv_mod2(phi)
            except linalg.linalg.LinAlgError:
                if verbose > 1:
                    print('Phi is still singular')
            else:
                if verbose:
                    print('Found a nonsingular phi on')
                    print('iterationCount = ', iterationCount)
                return [H_t, g, t]
        elif verbose > 1:
            print('phi is all zeros')
        iterationCount += 1
    if verbose:
        print('--- Error: nonsingular phi matrix not found.')

def inv_mod2(squareMatrix, verbose=0):
    if False:
        while True:
            i = 10
    '\n    Calculates the mod 2 inverse of a matrix.\n    '
    A = squareMatrix.copy()
    t = A.shape[0]
    if A.size == 1 and A[0] == 1:
        return array([1])
    Ainverse = inv(A)
    B = det(A) * Ainverse
    C = B % 2
    test = dot(A, C) % 2
    tempTest = zeros_like(test)
    for colNum in np.arange(test.shape[1]):
        for rowNum in np.arange(test.shape[0]):
            value = test[rowNum, colNum]
            if abs(1 - value) < 0.01:
                tempTest[rowNum, colNum] = 1
            elif abs(2 - value) < 0.01:
                tempTest[rowNum, colNum] = 0
            elif abs(0 - value) < 0.01:
                tempTest[rowNum, colNum] = 0
            elif verbose > 1:
                print('In inv_mod2. Rounding error on this')
                print('value? Mod 2 has already been done.')
                print('value:', value)
    test = tempTest.copy()
    if (test - eye(t, t) % 2).any():
        if verbose:
            print('Error in inv_mod2: did not find inverse.')
        raise linalg.linalg.LinAlgError
    else:
        return C

def swap_columns(a, b, arrayIn):
    if False:
        for i in range(10):
            print('nop')
    '\n    Swaps two columns in a matrix.\n    '
    arrayOut = arrayIn.copy()
    arrayOut[:, a] = arrayIn[:, b]
    arrayOut[:, b] = arrayIn[:, a]
    return arrayOut

def move_row_to_bottom(i, arrayIn):
    if False:
        for i in range(10):
            print('nop')
    '"\n    Moves a specified row (just one) to the bottom of the matrix,\n    then rotates the rows at the bottom up.\n\n    For example, if we had a matrix with 5 rows, and we wanted to\n    push row 2 to the bottom, then the resulting row order would be:\n    1,3,4,5,2\n    '
    arrayOut = arrayIn.copy()
    numRows = arrayOut.shape[0]
    arrayOut[numRows - 1] = arrayIn[i, :]
    index = 2
    while numRows - index >= i:
        arrayOut[numRows - index, :] = arrayIn[numRows - index + 1]
        index = index + 1
    return arrayOut

def get_full_rank_H_matrix(H, verbose=False):
    if False:
        return 10
    '\n    This function accepts a parity check matrix H and, if it is not\n    already full rank, will determine which rows are dependent and\n    remove them. The updated matrix will be returned.\n    '
    tempArray = H.copy()
    if linalg.matrix_rank(tempArray) == tempArray.shape[0]:
        if verbose:
            print('Returning H; it is already full rank.')
        return tempArray
    numRows = tempArray.shape[0]
    numColumns = tempArray.shape[1]
    limit = numRows
    rank = 0
    i = 0
    columnOrder = np.arange(numColumns).reshape(1, numColumns)
    rowOrder = np.arange(numRows).reshape(numRows, 1)
    while i < limit:
        if verbose:
            print('In get_full_rank_H_matrix; i:', i)
        found = False
        for j in np.arange(i, numColumns):
            if tempArray[i, j] == 1:
                found = True
                rank = rank + 1
                tempArray = swap_columns(j, i, tempArray)
                columnOrder = swap_columns(j, i, columnOrder)
                break
        if found == True:
            for k in np.arange(0, numRows):
                if k == i:
                    continue
                if tempArray[k, i] == 1:
                    tempArray[k, :] = tempArray[k, :] + tempArray[i, :]
                    tempArray = tempArray.copy() % 2
            i = i + 1
        if found == False:
            tempArray = move_row_to_bottom(i, tempArray)
            limit -= 1
            rowOrder = move_row_to_bottom(i, rowOrder)
    finalRowOrder = rowOrder[0:i]
    newNumberOfRowsForH = finalRowOrder.shape[0]
    newH = zeros((newNumberOfRowsForH, numColumns))
    for index in np.arange(newNumberOfRowsForH):
        newH[index, :] = H[finalRowOrder[index], :]
    tempHarray = newH.copy()
    for index in np.arange(numColumns):
        newH[:, index] = tempHarray[:, columnOrder[0, index]]
    if verbose:
        print('original H.shape:', H.shape)
        print('newH.shape:', newH.shape)
    return newH

def get_best_matrix(H, numIterations=100, verbose=0):
    if False:
        while True:
            i = 10
    '\n    This function will run the Greedy Upper Triangulation algorithm\n    for numIterations times, looking for the lowest possible gap.\n    The submatrices returned are those needed for real-time encoding.\n    '
    hadFirstJoy = 0
    index = 1
    while index <= numIterations:
        if verbose:
            print('--- In get_best_matrix, iteration:', index)
        index += 1
        try:
            ret = greedy_upper_triangulation(H, verbose)
        except ValueError as e:
            if verbose > 1:
                print('greedy_upper_triangulation error: ', e)
        else:
            if ret:
                [betterH, gap, t] = ret
            else:
                continue
            if not hadFirstJoy:
                hadFirstJoy = 1
                bestGap = gap
                bestH = betterH.copy()
                bestT = t
            elif gap < bestGap:
                bestGap = gap
                bestH = betterH.copy()
                bestT = t
    if hadFirstJoy:
        return [bestH, bestGap]
    else:
        if verbose:
            print('Error: Could not find appropriate H form')
            print('for encoding.')
        return

def getSystematicGmatrix(GenMatrix):
    if False:
        return 10
    '\n    This function finds the systematic form of the generator\n    matrix GenMatrix. This form is G = [I P] where I is an identity\n    matrix and P is the parity submatrix. If the GenMatrix matrix\n    provided is not full rank, then dependent rows will be deleted.\n\n    This function does not convert parity check (H) matrices to the\n    generator matrix format. Use the function getSystematicGmatrixFromH\n    for that purpose.\n    '
    tempArray = GenMatrix.copy()
    numRows = tempArray.shape[0]
    numColumns = tempArray.shape[1]
    limit = numRows
    rank = 0
    i = 0
    while i < limit:
        found = False
        for j in np.arange(i, numColumns):
            if tempArray[i, j] == 1:
                found = True
                rank = rank + 1
                tempArray = swap_columns(j, i, tempArray)
                break
        if found == True:
            for k in np.arange(0, numRows):
                if k == i:
                    continue
                if tempArray[k, i] == 1:
                    tempArray[k, :] = tempArray[k, :] + tempArray[i, :]
                    tempArray = tempArray.copy() % 2
            i = i + 1
        if found == False:
            tempArray = move_row_to_bottom(i, tempArray)
            limit -= 1
    G = tempArray[0:i, :]
    return G

def getSystematicGmatrixFromH(H, verbose=False):
    if False:
        print('Hello World!')
    '\n    If given a parity check matrix H, this function returns a\n    generator matrix G in the systematic form: G = [I P]\n      where:  I is an identity matrix, size k x k\n              P is the parity submatrix, size k x (n-k)\n    If the H matrix provided is not full rank, then dependent rows\n    will be deleted first.\n    '
    if verbose:
        print('received H with size: ', H.shape)
    tempArray = getSystematicGmatrix(H)
    n = H.shape[1]
    k = n - H.shape[0]
    I_temp = tempArray[:, 0:n - k]
    m = tempArray[:, n - k:n]
    newH = concatenate((m, I_temp), axis=1)
    k = m.shape[1]
    G = concatenate((identity(k), m.T), axis=1)
    if verbose:
        print('returning G with size: ', G.shape)
    return G