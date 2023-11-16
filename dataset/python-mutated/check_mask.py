import numpy as np

def check_mask(db, mask=(1, 0, 1)):
    if False:
        i = 10
        return i + 15
    out = np.zeros(db.shape[0], dtype=bool)
    for (idx, line) in enumerate(db):
        (target, vector) = (line[0], line[1:])
        if (mask == np.bitwise_and(mask, vector)).all():
            if target == 1:
                out[idx] = 1
    return out