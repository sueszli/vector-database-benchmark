"""
This is an auxiliary script that is used to compute valid PLL values to set
the CPU frequency to a given value.  The algorithm here appears as C code
for the machine.freq() function.
"""
from __future__ import print_function
import re

class MCU:

    def __init__(self, range_sysclk, range_m, range_n, range_p, range_q, range_vco_in, range_vco_out):
        if False:
            for i in range(10):
                print('nop')
        self.range_sysclk = range_sysclk
        self.range_m = range_m
        self.range_n = range_n
        self.range_p = range_p
        self.range_q = range_q
        self.range_vco_in = range_vco_in
        self.range_vco_out = range_vco_out
mcu_default = MCU(range_sysclk=range(2, 216 + 1, 2), range_m=range(2, 63 + 1), range_n=range(192, 432 + 1), range_p=range(2, 8 + 1, 2), range_q=range(2, 15 + 1), range_vco_in=range(1, 2 + 1), range_vco_out=range(192, 432 + 1))
mcu_table = {'stm32f413': MCU(range_sysclk=range(2, 100 + 1, 2), range_m=range(2, 63 + 1), range_n=range(50, 432 + 1), range_p=range(2, 8 + 1, 2), range_q=range(2, 15 + 1), range_vco_in=range(1, 2 + 1), range_vco_out=range(100, 432 + 1)), 'stm32h5': MCU(range_sysclk=range(2, 250 + 1, 2), range_m=range(1, 63 + 1), range_n=range(4, 512 + 1), range_p=range(2, 128 + 1, 2), range_q=range(1, 128 + 1), range_vco_in=range(1, 16 + 1), range_vco_out=range(150, 836 + 1)), 'stm32h7': MCU(range_sysclk=range(2, 400 + 1, 2), range_m=range(1, 63 + 1), range_n=range(4, 512 + 1), range_p=range(2, 128 + 1, 2), range_q=range(1, 128 + 1), range_vco_in=range(1, 16 + 1), range_vco_out=range(150, 960 + 1))}

def close_int(x):
    if False:
        print('Hello World!')
    return abs(x - round(x)) < 0.01

def compute_pll(hse, sys):
    if False:
        i = 10
        return i + 15
    for P in (2, 4, 6, 8):
        Q = sys * P / 48
        NbyM = sys * P / hse
        if not (close_int(NbyM) and close_int(Q)):
            continue
        if not 192 <= hse * NbyM <= 432:
            continue
        M = int(192 // NbyM)
        while hse > 2 * M or NbyM * M < 192:
            M += 1
        if not M <= hse:
            continue
        N = NbyM * M
        if not (192 <= N <= 432 and 2 <= Q <= 15):
            continue
        assert NbyM == N // M
        return (M, N, P, Q)
    return None

def compute_pll2(hse, sys, relax_pll48):
    if False:
        print('Hello World!')
    fallback = None
    for P in mcu.range_p:
        if sys * P not in mcu.range_vco_out:
            continue
        NbyM = float(sys * P) / hse
        M_min = mcu.range_n[0] // int(round(NbyM))
        while mcu.range_vco_in[-1] * M_min < hse:
            M_min += 1
        for M in range(M_min, hse + 1):
            N = NbyM * M
            if not close_int(N):
                continue
            N = round(N)
            if N not in mcu.range_n:
                continue
            Q = float(sys * P) / 48
            if close_int(Q) and round(Q) in mcu.range_q:
                return (M, N, P, Q)
            Q = (sys * P + 47) // 48
            if Q not in mcu.range_q:
                continue
            if fallback is None:
                fallback = (M, N, P, Q)
    if relax_pll48:
        return fallback
    else:
        return None

def compute_derived(hse, pll):
    if False:
        print('Hello World!')
    hse = float(hse)
    (M, N, P, Q) = pll
    vco_in = hse / M
    vco_out = hse * N / M
    pllck = hse / M * N / P
    pll48ck = hse / M * N / Q
    return (vco_in, vco_out, pllck, pll48ck)

def verify_pll(hse, pll):
    if False:
        while True:
            i = 10
    (M, N, P, Q) = pll
    (vco_in, vco_out, pllck, pll48ck) = compute_derived(hse, pll)
    assert close_int(M)
    assert close_int(N)
    assert close_int(P)
    assert close_int(Q)
    assert M in mcu.range_m
    assert N in mcu.range_n
    assert P in mcu.range_p
    assert Q in mcu.range_q
    assert mcu.range_vco_in[0] <= vco_in <= mcu.range_vco_in[-1]
    assert mcu.range_vco_out[0] <= vco_out <= mcu.range_vco_out[-1]

def compute_pll_table(source_clk, relax_pll48):
    if False:
        for i in range(10):
            print('nop')
    valid_plls = []
    for sysclk in mcu.range_sysclk:
        pll = compute_pll2(source_clk, sysclk, relax_pll48)
        if pll is not None:
            verify_pll(source_clk, pll)
            valid_plls.append((sysclk, pll))
    return valid_plls

def generate_c_table(hse, valid_plls):
    if False:
        i = 10
        return i + 15
    valid_plls.sort()
    if mcu.range_sysclk[-1] <= 255 and mcu.range_m[-1] <= 63 and (mcu.range_p[-1] // 2 - 1 <= 3):
        typedef = 'uint16_t'
        sys_mask = 255
        m_shift = 10
        m_mask = 63
        p_shift = 8
        p_mask = 3
    else:
        typedef = 'uint32_t'
        sys_mask = 65535
        m_shift = 24
        m_mask = 255
        p_shift = 16
        p_mask = 255
    print('#define PLL_FREQ_TABLE_SYS(pll) ((pll) & %d)' % (sys_mask,))
    print('#define PLL_FREQ_TABLE_M(pll) (((pll) >> %d) & %d)' % (m_shift, m_mask))
    print('#define PLL_FREQ_TABLE_P(pll) (((((pll) >> %d) & %d) + 1) * 2)' % (p_shift, p_mask))
    print('typedef %s pll_freq_table_t;' % (typedef,))
    print('// (M, P/2-1, SYS) values for %u MHz source' % hse)
    print('static const pll_freq_table_t pll_freq_table[%u] = {' % (len(valid_plls),))
    for (sys, (M, N, P, Q)) in valid_plls:
        print('    (%u << %u) | (%u << %u) | %u,' % (M, m_shift, P // 2 - 1, p_shift, sys), end='')
        if M >= 2:
            (vco_in, vco_out, pllck, pll48ck) = compute_derived(hse, (M, N, P, Q))
            print(' // M=%u N=%u P=%u Q=%u vco_in=%.2f vco_out=%.2f pll48=%.2f' % (M, N, P, Q, vco_in, vco_out, pll48ck), end='')
        print()
    print('};')

def print_table(hse, valid_plls):
    if False:
        while True:
            i = 10
    print('HSE =', hse, 'MHz')
    print('sys :  M      N     P     Q : VCO_IN VCO_OUT   PLLCK PLL48CK')
    out_format = '%3u : %2u  %.1f  %.2f  %.2f :  %5.2f  %6.2f  %6.2f  %6.2f'
    for (sys, pll) in valid_plls:
        print(out_format % ((sys,) + pll + compute_derived(hse, pll)))
    print('found %u valid configurations' % len(valid_plls))

def search_header_for_hsx_values(filename, vals):
    if False:
        while True:
            i = 10
    regex_inc = re.compile('#include "(boards/[A-Za-z0-9_./]+)"')
    regex_def = re.compile('#define +(HSE_VALUE|HSI_VALUE) +\\((\\(uint32_t\\))?([0-9]+)\\)')
    with open(filename) as f:
        for line in f:
            line = line.strip()
            m = regex_inc.match(line)
            if m:
                search_header_for_hsx_values(m.group(1), vals)
                continue
            m = regex_def.match(line)
            if m:
                val = int(m.group(3)) // 1000000
                if m.group(1) == 'HSE_VALUE':
                    vals[0] = val
                else:
                    vals[1] = val
    return vals

def main():
    if False:
        while True:
            i = 10
    global mcu
    global out_format
    import sys
    argv = sys.argv[1:]
    c_table = False
    mcu_series = 'stm32f4'
    hse = None
    hsi = None
    while True:
        if argv[0] == '-c':
            c_table = True
            argv.pop(0)
        elif argv[0] == '-m':
            argv.pop(0)
            mcu_series = argv.pop(0).lower()
        else:
            break
    if len(argv) != 1:
        print('usage: pllvalues.py [-c] [-m <mcu_series>] <hse in MHz>')
        sys.exit(1)
    if argv[0].startswith('file:'):
        (hse, hsi) = search_header_for_hsx_values(argv[0][5:], [None, None])
        if hse is None:
            raise ValueError('%s does not contain a definition of HSE_VALUE' % argv[0])
    else:
        hse = int(argv[0])
    mcu = mcu_default
    for m in mcu_table:
        if mcu_series.startswith(m):
            mcu = mcu_table[m]
            break
    relax_pll48 = mcu_series.startswith(('stm32f413', 'stm32f7', 'stm32h5', 'stm32h7'))
    hse_valid_plls = compute_pll_table(hse, relax_pll48)
    if hsi is not None:
        hsi_valid_plls = compute_pll_table(hsi, relax_pll48)
    if c_table:
        print('#if MICROPY_HW_CLK_USE_HSI')
        if hsi is not None:
            hsi_valid_plls.append((hsi, (0, 0, 2, 0)))
            generate_c_table(hsi, hsi_valid_plls)
        print('#else')
        if hsi is not None:
            hse_valid_plls.append((hsi, (0, 0, 2, 0)))
        hse_valid_plls.append((hse, (1, 0, 2, 0)))
        generate_c_table(hse, hse_valid_plls)
        print('#endif')
    else:
        print_table(hse, hse_valid_plls)
if __name__ == '__main__':
    main()