import math

def window_floor(idx, radius):
    if False:
        i = 10
        return i + 15
    if radius > idx:
        return 0
    else:
        return idx - radius

def window_ceil(idx, ceil, radius):
    if False:
        for i in range(10):
            print('nop')
    if idx + radius > ceil:
        return ceil
    else:
        return idx + radius

def growcut(image, state, state_next, window_radius):
    if False:
        while True:
            i = 10
    changes = 0
    sqrt_3 = math.sqrt(3.0)
    height = image.shape[0]
    width = image.shape[1]
    for j in range(width):
        for i in range(height):
            winning_colony = state[i, j, 0]
            defense_strength = state[i, j, 1]
            for jj in range(window_floor(j, window_radius), window_ceil(j + 1, width, window_radius)):
                for ii in range(window_floor(i, window_radius), window_ceil(i + 1, height, window_radius)):
                    if ii != i and jj != j:
                        d = image[i, j, 0] - image[ii, jj, 0]
                        s = d * d
                        for k in range(1, 3):
                            d = image[i, j, k] - image[ii, jj, k]
                            s += d * d
                        gval = 1.0 - math.sqrt(s) / sqrt_3
                        attack_strength = gval * state[ii, jj, 1]
                        if attack_strength > defense_strength:
                            defense_strength = attack_strength
                            winning_colony = state[ii, jj, 0]
                            changes += 1
            state_next[i, j, 0] = winning_colony
            state_next[i, j, 1] = defense_strength
    return changes