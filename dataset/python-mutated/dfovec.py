import numpy as np

def dfovec(m, n, x, nprob):
    if False:
        while True:
            i = 10
    c13 = 13.0
    c14 = 14.0
    c29 = 29.0
    c45 = 45.0
    v = [4.0, 2.0, 1.0, 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0625]
    y1 = [0.14, 0.18, 0.22, 0.25, 0.29, 0.32, 0.35, 0.39, 0.37, 0.58, 0.73, 0.96, 1.34, 2.1, 4.39]
    y2 = [0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246]
    y3 = [34780.0, 28610.0, 23650.0, 19630.0, 16370.0, 13720.0, 11540.0, 9744.0, 8261.0, 7030.0, 6005.0, 5147.0, 4427.0, 3820.0, 3307.0, 2872.0]
    y4 = [0.844, 0.908, 0.932, 0.936, 0.925, 0.908, 0.881, 0.85, 0.818, 0.784, 0.751, 0.718, 0.685, 0.658, 0.628, 0.603, 0.58, 0.558, 0.538, 0.522, 0.506, 0.49, 0.478, 0.467, 0.457, 0.448, 0.438, 0.431, 0.424, 0.42, 0.414, 0.411, 0.406]
    y5 = [1.366, 1.191, 1.112, 1.013, 0.991, 0.885, 0.831, 0.847, 0.786, 0.725, 0.746, 0.679, 0.608, 0.655, 0.616, 0.606, 0.602, 0.626, 0.651, 0.724, 0.649, 0.649, 0.694, 0.644, 0.624, 0.661, 0.612, 0.558, 0.533, 0.495, 0.5, 0.423, 0.395, 0.375, 0.372, 0.391, 0.396, 0.405, 0.428, 0.429, 0.523, 0.562, 0.607, 0.653, 0.672, 0.708, 0.633, 0.668, 0.645, 0.632, 0.591, 0.559, 0.597, 0.625, 0.739, 0.71, 0.729, 0.72, 0.636, 0.581, 0.428, 0.292, 0.162, 0.098, 0.054]
    fvec = np.zeros(m)
    total = 0
    if nprob == 1:
        for j in range(n):
            total = total + x[j]
        temp = 2 * total / m + 1
        for i in range(m):
            fvec[i] = -temp
            if i < n:
                fvec[i] = fvec[i] + x[i]
    elif nprob == 2:
        for j in range(n):
            total = total + (j + 1) * x[j]
        for i in range(m):
            fvec[i] = (i + 1) * total - 1
    elif nprob == 3:
        for j in range(1, n - 1):
            total = total + (j + 1) * x[j]
        for i in range(m - 1):
            fvec[i] = i * total - 1
        fvec[m - 1] = -1
    elif nprob == 4:
        fvec[0] = 10 * (x[1] - x[0] * x[0])
        fvec[1] = 1 - x[0]
    elif nprob == 5:
        if x[0] > 0:
            th = np.arctan(x[1] / x[0]) / (2 * np.pi)
        elif x[0] < 0:
            th = np.arctan(x[1] / x[0]) / (2 * np.pi) + 0.5
        elif x[0] == x[1] and x[1] == 0:
            th = 0.0
        else:
            th = 0.25
        r = np.sqrt(x[0] * x[0] + x[1] * x[1])
        fvec[0] = 10 * (x[2] - 10 * th)
        fvec[1] = 10 * (r - 1)
        fvec[2] = x[2]
    elif nprob == 6:
        fvec[0] = x[0] + 10 * x[1]
        fvec[1] = np.sqrt(5) * (x[2] - x[3])
        fvec[2] = (x[1] - 2 * x[2]) ** 2
        fvec[3] = np.sqrt(10) * (x[0] - x[3]) ** 2
    elif nprob == 7:
        fvec[0] = -c13 + x[0] + ((5 - x[1]) * x[1] - 2) * x[1]
        fvec[1] = -c29 + x[0] + ((1 + x[1]) * x[1] - c14) * x[1]
    elif nprob == 8:
        for i in range(15):
            tmp1 = i + 1
            tmp2 = 15 - i
            tmp3 = tmp1
            if i > 7:
                tmp3 = tmp2
            fvec[i] = y1[i] - (x[0] + tmp1 / (x[1] * tmp2 + x[2] * tmp3))
    elif nprob == 9:
        for i in range(11):
            tmp1 = v[i] * (v[i] + x[1])
            tmp2 = v[i] * (v[i] + x[2]) + x[3]
            fvec[i] = y2[i] - x[0] * tmp1 / tmp2
    elif nprob == 10:
        for i in range(16):
            temp = 5 * (i + 1) + c45 + x[2]
            tmp1 = x[1] / temp
            tmp2 = np.exp(tmp1)
            fvec[i] = x[0] * tmp2 - y3[i]
    elif nprob == 11:
        for i in range(29):
            div = (i + 1) / c29
            s1 = 0
            dx = 1
            for j in range(1, n):
                s1 = s1 + j * dx * x[j]
                dx = div * dx
            s2 = 0
            dx = 1
            for j in range(n):
                s2 = s2 + dx * x[j]
                dx = div * dx
            fvec[i] = s1 - s2 * s2 - 1
        fvec[29] = x[0]
        fvec[30] = x[1] - x[0] * x[0] - 1
    elif nprob == 12:
        for i in range(m):
            temp = i + 1
            tmp1 = temp / 10
            fvec[i] = np.exp(-tmp1 * x[0]) - np.exp(-tmp1 * x[1]) + (np.exp(-temp) - np.exp(-tmp1)) * x[2]
    elif nprob == 13:
        for i in range(m):
            temp = i + 1
            fvec[i] = 2 + 2 * temp - np.exp(temp * x[0]) - np.exp(temp * x[1])
    elif nprob == 14:
        for i in range(m):
            temp = (i + 1) / 5
            tmp1 = x[0] + temp * x[1] - np.exp(temp)
            tmp2 = x[2] + np.sin(temp) * x[3] - np.cos(temp)
            fvec[i] = tmp1 * tmp1 + tmp2 * tmp2
    elif nprob == 15:
        for j in range(n):
            t1 = 1
            t2 = 2 * x[j] - 1
            t = 2 * t2
            for i in range(m):
                fvec[i] = fvec[i] + t2
                th = t * t2 - t1
                t1 = t2
                t2 = th
        iev = -1
        for i in range(m):
            fvec[i] = fvec[i] / n
            if iev > 0:
                fvec[i] = fvec[i] + 1 / ((i + 1) ** 2 - 1)
            iev = -iev
    elif nprob == 16:
        total1 = -(n + 1)
        prod1 = 1
        for j in range(n):
            total1 = total1 + x[j]
            prod1 = x[j] * prod1
        for i in range(n - 1):
            fvec[i] = x[i] + total1
        fvec[n - 1] = prod1 - 1
    elif nprob == 17:
        for i in range(33):
            temp = 10 * i
            tmp1 = np.exp(-x[3] * temp)
            tmp2 = np.exp(-x[4] * temp)
            fvec[i] = y4[i] - (x[0] + x[1] * tmp1 + x[2] * tmp2)
    elif nprob == 18:
        for i in range(65):
            temp = i / 10
            tmp1 = np.exp(-x[4] * temp)
            tmp2 = np.exp(-x[5] * (temp - x[8]) ** 2)
            tmp3 = np.exp(-x[6] * (temp - x[9]) ** 2)
            tmp4 = np.exp(-x[7] * (temp - x[10]) ** 2)
            fvec[i] = y5[i] - (x[0] * tmp1 + x[1] * tmp2 + x[2] * tmp3 + x[3] * tmp4)
    elif nprob == 19:
        for i in range(n - 4):
            fvec[i] = -4 * x[i] + 3.0
            fvec[n - 4 + i] = x[i] ** 2 + 2 * x[i + 1] ** 2 + 3 * x[i + 2] ** 2 + 4 * x[i + 3] ** 2 + 5 * x[n - 1] ** 2
    elif nprob == 20:
        fvec[0] = x[0] - 1.0
        for i in range(1, n):
            fvec[i] = 10 * (x[i] - x[i - 1] ** 3)
    elif nprob == 21:
        for i in range(n):
            ss = 0
            for j in range(n):
                v2 = np.sqrt(x[i] ** 2 + (i + 1) / (j + 1))
                ss = ss + v2 * (np.sin(np.log(v2)) ** 5 + np.cos(np.log(v2)) ** 5)
            fvec[i] = 1400 * x[i] + (i - 49) ** 3 + ss
    elif nprob == 22:
        fvec[0] = x[0] + x[1] + 0.69
        fvec[1] = x[2] + x[3] + 0.044
        fvec[2] = x[4] * x[0] + x[5] * x[1] - x[6] * x[2] - x[7] * x[3] + 1.57
        fvec[3] = x[6] * x[0] + x[7] * x[1] + x[4] * x[2] + x[5] * x[3] + 1.31
        fvec[4] = x[0] * (x[4] ** 2 - x[6] ** 2) - 2.0 * x[2] * x[4] * x[6] + x[1] * (x[5] ** 2 - x[7] ** 2) - 2.0 * x[3] * x[5] * x[7] + 2.65
        fvec[5] = x[2] * (x[4] ** 2 - x[6] ** 2) + 2.0 * x[0] * x[4] * x[6] + x[3] * (x[5] ** 2 - x[7] ** 2) + 2.0 * x[1] * x[5] * x[7] - 2.0
        fvec[6] = x[0] * x[4] * (x[4] ** 2 - 3.0 * x[6] ** 2) + x[2] * x[6] * (x[6] ** 2 - 3.0 * x[4] ** 2) + x[1] * x[5] * (x[5] ** 2 - 3.0 * x[7] ** 2) + x[3] * x[7] * (x[7] ** 2 - 3.0 * x[5] ** 2) + 12.6
        fvec[7] = x[2] * x[4] * (x[4] ** 2 - 3.0 * x[6] ** 2) - x[0] * x[6] * (x[6] ** 2 - 3.0 * x[4] ** 2) + x[3] * x[5] * (x[5] ** 2 - 3.0 * x[7] ** 2) - x[1] * x[7] * (x[7] ** 2 - 3.0 * x[6] ** 2) - 9.48
    else:
        print(f'unrecognized function number {nprob}')
        return None
    return fvec