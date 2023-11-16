def specialconvolve(a):
    if False:
        for i in range(10):
            print('nop')
    rowconvol = a[1:-1, :] + a[:-2, :] + a[2:, :]
    colconvol = rowconvol[:, 1:-1] + rowconvol[:, :-2] + rowconvol[:, 2:] - 9 * a[1:-1, 1:-1]
    return colconvol