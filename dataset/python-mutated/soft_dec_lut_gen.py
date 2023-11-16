import numpy

def soft_dec_table_generator(soft_dec_gen, prec, Es=1):
    if False:
        return 10
    "\n    | Builds a LUT that is a list of tuples. The tuple represents the\n    | soft decisions for the constellation/bit mapping at any given\n    | point in the complex space, (x,y).\n    |\n    | The table is built to a precision specified by the 'prec'\n    | argument. There are (2x2)^prec samples in the sample space, so we\n    | get the precision of 2^prec samples in both the real and imaginary\n    | axes.\n    |\n    | The space is represented where index 0 is the bottom left corner\n    | and the maximum index is the upper left. The table index for a\n    | surface space with 4 bits of precision looks like the following:\n    |\n    | 240 241 242 243 244 245 246 247 | 248 249 250 251 252 253 254 255\n    | 224 225 226 227 228 229 230 231 | 232 233 234 235 236 237 238 239\n    | 208 209 210 211 212 213 214 215 | 216 217 218 219 220 221 222 223\n    | 192 193 194 195 196 197 198 199 | 200 201 202 203 204 205 206 207\n    | 176 177 178 179 180 181 182 183 | 184 185 186 187 188 189 190 191\n    | 160 161 162 163 164 165 166 167 | 168 169 170 171 172 173 174 175\n    | 144 145 146 147 148 149 150 151 | 152 153 154 155 156 157 158 159\n    | 128 129 130 131 132 133 134 135 | 136 137 138 139 140 141 142 143\n    | -----------------------------------------------------------------\n    | 112 113 114 115 116 117 118 119 | 120 121 122 123 124 125 126 127\n    |  96  97  98  99 100 101 102 103 | 104 105 106 107 108 109 110 111\n    |  80  81  82  83  84  85  86  87 |  88  89  90  91  92  93  94  95\n    |  64  65  66  67  68  69  70  71 |  72  73  74  75  76  77  78  79\n    |  48  49  50  51  52  53  54  55 |  56  57  58  59  60  61  62  63\n    |  32  33  34  35  36  37  38  39 |  40  41  42  43  44  45  46  47\n    |  16  17  18  19  20  21  22  23 |  24  25  26  27  28  29  30  31\n    |   0   1   2   3   4   5   6   7 |   8   9  10  11  12  13  14  15\n    |\n    | We then calculate coordinates from -1 to 1 with 2^prec points for\n    | both the x and y axes. We then sample starting at (-1, -1) and\n    | move left to right on the x-axis and then move up a row on the\n    | y-axis. For every point in this sampled space, we calculate the\n    | soft decisions for the given constellation/mapping. This is done\n    | by passing in the function 'soft_dec_gen' as an argument to this\n    | function. This takes in the x/y coordinates and outputs the soft\n    | decisions. These soft decisions are stored into the list at the\n    | index from the above table as a tuple.\n    |\n    | The function 'calc_from_table' takes in a point and reverses this\n    | operation. It converts the point from the coordinates (-1,-1) to\n    | (1,1) into an index value in the table and returns the tuple of\n    | soft decisions at that index.\n    |\n    | Es is the maximum energy per symbol. This is passed to the\n    | function to provide the bounds when calling the generator function\n    | since they don't know how the constellation was normalized. Using\n    | the (maximum) energy per symbol for constellation allows us to\n    | provide any scaling of the constellation (normalized to sum to 1,\n    | normalized so the outside points sit on +/-1, etc.) but still\n    | calculate the soft decisions as we would given the full\n    | constellation.\n\n    "
    npts = int(2.0 ** prec)
    maxd = Es * numpy.sqrt(2.0) / 2.0
    yrng = numpy.linspace(-maxd, maxd, npts)
    xrng = numpy.linspace(-maxd, maxd, npts)
    table = []
    for y in yrng:
        for x in xrng:
            pt = complex(x, y)
            decs = soft_dec_gen(pt, Es)
            table.append(decs)
    return table

def const_normalization(constel, method='POWER'):
    if False:
        for i in range(10):
            print('nop')
    constel = numpy.array(constel)
    if method == 'POWER':
        power = 0
        for point in constel:
            power += numpy.abs(point) ** 2
        amplitude = numpy.sqrt(power / len(constel))
        constel = constel / amplitude
    return constel

def min_max_axes(constel):
    if False:
        print('Hello World!')
    constel = numpy.array(constel)
    re_min = min(constel.real)
    im_min = min(constel.imag)
    re_max = max(constel.real)
    im_max = max(constel.imag)
    max_amp = numpy.float32(max([abs(re_min), abs(im_min), re_max, im_max]))
    return max_amp

def soft_dec_table(constel, symbols, prec, npwr=1):
    if False:
        while True:
            i = 10
    '\n    Similar in nature to soft_dec_table_generator above. Instead, this\n    takes in the constellation and symbol points along with the noise\n    power estimate and uses calc_soft_dec (below) to generate the\n    LUT.\n\n    Instead of assuming that the constellation is normalied (e.g., all\n    points are between -1 and 1), this function calculates the min/max\n    of both the real and imaginary axes and uses those when\n    constructing the LUT. So when using this version of the LUT, the\n    samples and the constellations must be working on the same\n    magnitudes.\n\n    Because this uses the calc_soft_dec function, it can be quite\n    a bit more expensive to generate the LUT, though it should be\n    one-time work.\n    '
    padding = numpy.float32(2.0)
    constel = const_normalization(constel, 'POWER')
    max_amp = min_max_axes(constel)
    d_lut_scale = numpy.float32(2.0 ** int(prec))
    border = numpy.float32(1.0 / d_lut_scale)
    maxd = numpy.float32(max_amp * padding - border)
    step = numpy.float32(2.0 * maxd / (d_lut_scale - 2.0))
    table = []
    endstop = int(d_lut_scale - 2)
    start = -maxd + step / 2
    y = 0
    while y < endstop:
        x = 0
        if y == 0:
            while x < endstop:
                if x == 0 or x == endstop - 1:
                    pt = complex(start + x * step, start + y * step)
                    table.append(calc_soft_dec(pt, constel, symbols, npwr))
                pt = complex(start + x * step, start + y * step)
                table.append(calc_soft_dec(pt, constel, symbols, npwr))
                x += 1
            x = 0
        while x < endstop:
            if x == 0 or x == endstop - 1:
                pt = complex(start + x * step, start + y * step)
                table.append(calc_soft_dec(pt, constel, symbols, npwr))
            pt = complex(start + x * step, start + y * step)
            table.append(calc_soft_dec(pt, constel, symbols, npwr))
            x += 1
        x = 0
        if y == endstop - 1:
            while x < endstop:
                if x == 0 or x == endstop - 1:
                    pt = complex(start + x * step, start + y * step)
                    table.append(calc_soft_dec(pt, constel, symbols, npwr))
                pt = complex(start + x * step, start + y * step)
                table.append(calc_soft_dec(pt, constel, symbols, npwr))
                x += 1
        y += 1
    return table

def calc_soft_dec_from_table(sample, table, prec: numpy.float32, d_maxamp=1):
    if False:
        while True:
            i = 10
    "\n    Takes in a complex sample and converts it from the coordinates\n    (-1,-1) to (1,1) into an index value. The index value points to a\n    location in the provided LUT 'table' and returns the soft\n    decisions tuple at that index.\n\n    sample: the complex sample to calculate the soft decisions\n    from.\n\n    table: the LUT.\n\n    prec: the precision used when generating the LUT.\n\n    Es: the energy per symbol. This is passed to the function to\n    provide the bounds when calling the generator function since they\n    don't know how the constellation was normalized. Using the\n    (maximum) energy per symbol for constellation allows us to provide\n    any scaling of the constellation (normalized to sum to 1,\n    normalized so the outside points sit on +/-1, etc.) but still\n    calculate the soft decisions as we would given the full\n    constellation.\n    "
    d_lut_scale = int(2 ** prec)
    d_padding = 2.0
    border = numpy.float32(1.0 / d_lut_scale)
    maxd = numpy.float32(d_padding * d_maxamp)
    limit = numpy.float32(maxd - border)
    xre = numpy.float32(branchless_clip(numpy.float32(sample.real), limit) / maxd)
    xim = numpy.float32(branchless_clip(numpy.float32(sample.imag), limit) / maxd)
    scale = numpy.float32((d_lut_scale - 2.0) / 2.0)
    xre = numpy.floor(numpy.float32((1.0 + xre) * scale)) + 1
    xim = numpy.floor(numpy.float32((1.0 + xim) * scale)) + 1
    point = table_lookup(xre, xim, d_lut_scale, table)
    return point

def table_lookup(xre, xim, d_lut_scale, table):
    if False:
        i = 10
        return i + 15
    index = int(numpy.float32(d_lut_scale * xim + xre))
    max_index = d_lut_scale * d_lut_scale
    while index >= max_index:
        index -= d_lut_scale
    while index < 0:
        index += d_lut_scale
    return table[index]

def branchless_clip(x, clip):
    if False:
        i = 10
        return i + 15
    return numpy.float32(0.5 * (abs(x + clip) - abs(x - clip)))

def calc_soft_dec(sample, constel, symbols, npwr=1):
    if False:
        while True:
            i = 10
    "\n    This function takes in any constellation and symbol set\n    (where symbols[i] is the set of bits at constellation point\n    constel[i] and an estimate of the noise power and produces the\n    soft decisions for the given sample.\n\n    If known, the noise power of the received sample may be passed in\n    to this function as npwr.\n\n    This is an incredibly costly algorithm because it must calculate\n    the Euclidean distance between the sample and all points in the\n    constellation to build up its probability\n    calculations. Conversely, it should work for any given\n    constellation/symbol map.\n\n    The function returns a vector of k soft decisions. Decisions less\n    than 0 are more likely to indicate a '0' bit and decisions greater\n    than 0 are more likely to indicate a '1' bit.\n    "
    M = len(constel)
    k = int(numpy.log2(M))
    tmp = 2 * k * [0]
    s = k * [0]
    for i in range(M):
        dist = abs(sample - constel[i]) ** 2
        arg = -dist / (npwr * 1.0)
        argmin = -86
        if arg < argmin:
            d = 3.84745e-36 / -arg
        else:
            d = numpy.exp(arg)
        for j in range(k):
            mask = 1 << j
            bit = (symbols[i] & mask) >> j
            if bit == 0:
                tmp[2 * j + 0] += d
            else:
                tmp[2 * j + 1] += d
    for i in range(k):
        one = tmp[2 * i + 1]
        zero = tmp[2 * i + 0]
        if one < 1e-45:
            one = 1e-45
        if zero < 1e-45:
            zero = 1e-45
        s[k - 1 - i] = numpy.log(one) - numpy.log(zero)
    return s

def show_table(table):
    if False:
        print('Hello World!')
    prec = int(numpy.sqrt(len(table)))
    nbits = len(table[0])
    pp = ''
    subi = 1
    subj = 0
    for i in reversed(list(range(prec + 1))):
        if i == prec // 2:
            pp += '-----' + prec * (nbits * 8 + 3) * '-' + '\n'
            subi = 0
            continue
        for j in range(prec + 1):
            if j == prec // 2:
                pp += '| '
                subj = 1
            else:
                item = table[prec * (i - subi) + (j - subj)]
                pp += '( '
                for t in range(nbits - 1, -1, -1):
                    pp += '{0: .4f} '.format(item[t])
                pp += ') '
        pp += '\n'
        subj = 0
    print(pp)