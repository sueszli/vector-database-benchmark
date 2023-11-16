def within_block_loop(R, T, start_stop_pairs, nblocks):
    if False:
        print('Hello World!')
    for (start, stop) in start_stop_pairs:
        for j in range(start, stop):
            for i in range(j - 1, start - 1, -1):
                s = 0
                if j - i > 1:
                    for k in range(i + 1, j):
                        s += R[i, k] * R[k, j]
                denom = R[i, i] + R[j, j]
                num = T[i, j] - s
                if denom != 0:
                    R[i, j] = (T[i, j] - s) / denom
                elif denom == 0 and num == 0:
                    R[i, j] = 0
                else:
                    raise RuntimeError('failed to find the matrix square root')