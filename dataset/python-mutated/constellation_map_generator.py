def constellation_map_generator(basis_cpoints, basis_symbols, k, pi):
    if False:
        print('Hello World!')
    '\n    Uses the a basis constellation provided (e.g., from\n    psk_constellation.psk_4()) and the the k and permutation index\n    (pi) to generate a new Gray-coded symbol map to the constellation\n    points provided in the basis.\n\n    The basis_cpoints are the constellation points of the basis\n    constellation, and basis_symbols are the symbols that correspond\n    to the constellation points.\n\n    The selection of k and pi will provide an automorphism the\n    hyperoctahedral group of the basis constellation.\n\n    This function returns a tuple of (constellation_points,\n    symbol_map). The constellation_points is a list of the\n    constellation points in complex space and the symbol_map is a list\n    of the log2(M)-bit symbols for the constellation points (i.e.,\n    symbol_map[i] are the bits associated with\n    constellation_points[i]).\n    '
    const_points = basis_cpoints
    s = basis_symbols
    symbols = list()
    for s_i in s:
        tmp = 0
        for (i, p) in enumerate(pi):
            bit = s_i >> i & 1
            tmp |= bit << p
        symbols.append(tmp ^ k)
    return (const_points, symbols)