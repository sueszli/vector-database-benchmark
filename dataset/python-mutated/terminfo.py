import re
from binascii import hexlify, unhexlify
from typing import Dict, Generator, Optional, cast
from kitty.options.types import Options

def modify_key_bytes(keybytes: bytes, amt: int) -> bytes:
    if False:
        print('Hello World!')
    if amt == 0:
        return keybytes
    ans = bytearray(keybytes)
    samt = str(amt).encode('ascii')
    if ans[-1] == ord('~'):
        return bytes(ans[:-1] + bytearray(b';' + samt + b'~'))
    if ans[1] == ord('O'):
        return bytes(ans[:1] + bytearray(b'[1;' + samt) + ans[-1:])
    raise ValueError(f'Unknown key type in key: {keybytes!r}')

def encode_keystring(keybytes: bytes) -> str:
    if False:
        return 10
    return keybytes.decode('ascii').replace('\x1b', '\\E')
names = (Options.term, 'KovIdTTY')
termcap_aliases = {'TN': 'name'}
bool_capabilities = {'am', 'ccc', 'km', 'mc5i', 'mir', 'msgr', 'npc', 'xenl', 'hs', 'Tc', 'Su', 'fullkbd', 'XF'}
termcap_aliases.update({'am': 'am', 'cc': 'ccc', 'km': 'km', '5i': 'mc5i', 'mi': 'mir', 'ms': 'msgr', 'NP': 'npc', 'xn': 'xenl', 'hs': 'hs'})
numeric_capabilities = {'colors': 256, 'cols': 80, 'lines': 24, 'it': 8, 'pairs': 32767}
termcap_aliases.update({'Co': 'colors', 'pa': 'pairs', 'li': 'lines', 'co': 'cols', 'it': 'it'})
string_capabilities = {'acsc': '++\\,\\,--..00``aaffgghhiijjkkllmmnnooppqqrrssttuuvvwwxxyyzz{{||}}~~', 'bel': '^G', 'bold': '\\E[1m', 'cbt': '\\E[Z', 'kcbt': '\\E[Z', 'civis': '\\E[?25l', 'clear': '\\E[H\\E[2J', 'cnorm': '\\E[?12h\\E[?25h', 'cr': '^M', 'csr': '\\E[%i%p1%d;%p2%dr', 'cub': '\\E[%p1%dD', 'cub1': '^H', 'cud': '\\E[%p1%dB', 'cud1': '^J', 'cuf': '\\E[%p1%dC', 'cuf1': '\\E[C', 'cuu': '\\E[%p1%dA', 'cuu1': '\\E[A', 'cup': '\\E[%i%p1%d;%p2%dH', 'cvvis': '\\E[?12;25h', 'dch': '\\E[%p1%dP', 'dch1': '\\E[P', 'dim': '\\E[2m', 'dl': '\\E[%p1%dM', 'dl1': '\\E[M', 'ech': '\\E[%p1%dX', 'ed': '\\E[J', 'el': '\\E[K', 'el1': '\\E[1K', 'flash': '\\E[?5h$<100/>\\E[?5l', 'home': '\\E[H', 'hpa': '\\E[%i%p1%dG', 'ht': '^I', 'hts': '\\EH', 'ich': '\\E[%p1%d@', 'il': '\\E[%p1%dL', 'il1': '\\E[L', 'ind': '^J', 'indn': '\\E[%p1%dS', 'initc': '\\E]4;%p1%d;rgb\\:%p2%{255}%*%{1000}%/%2.2X/%p3%{255}%*%{1000}%/%2.2X/%p4%{255}%*%{1000}%/%2.2X\\E\\\\', 'oc': '\\E]104\\007', 'kbs': '\\177', 'kmous': '\\E[M', 'kri': '\\E[1;2A', 'kind': '\\E[1;2B', 'rc': '\\E8', 'rep': '%p1%c\\E[%p2%{1}%-%db', 'rev': '\\E[7m', 'ri': '\\EM', 'rin': '\\E[%p1%dT', 'rmam': '\\E[?7l', 'rmcup': '\\E[?1049l', 'rmir': '\\E[4l', 'rmkx': '\\E[?1l', 'rmso': '\\E[27m', 'rmul': '\\E[24m', 'rmxx': '\\E[29m', 'rs1': '\\E]\\E\\\\\\Ec', 'sc': '\\E7', 'setab': '\\E[%?%p1%{8}%<%t4%p1%d%e%p1%{16}%<%t10%p1%{8}%-%d%e48;5;%p1%d%;m', 'setaf': '\\E[%?%p1%{8}%<%t3%p1%d%e%p1%{16}%<%t9%p1%{8}%-%d%e38;5;%p1%d%;m', 'sgr': '%?%p9%t\\E(0%e\\E(B%;\\E[0%?%p6%t;1%;%?%p2%t;4%;%?%p1%p3%|%t;7%;%?%p4%t;5%;%?%p7%t;8%;m', 'sgr0': '\\E(B\\E[m', 'op': '\\E[39;49m', 'smam': '\\E[?7h', 'smcup': '\\E[?1049h', 'smir': '\\E[4h', 'smkx': '\\E[?1h', 'smso': '\\E[7m', 'smul': '\\E[4m', 'Smulx': '\\E[4:%p1%dm', 'smxx': '\\E[9m', 'Sync': '\\EP=%p1%ds\\E\\\\', 'tbc': '\\E[3g', 'tsl': '\\E]2;', 'fsl': '^G', 'dsl': '\\E]2;\\E\\\\', 'vpa': '\\E[%i%p1%dd', 'sitm': '\\E[3m', 'ritm': '\\E[23m', 'smacs': '\\E(0', 'rmacs': '\\E(B', 'khlp': '', 'kund': '', 'ka1': '', 'ka3': '', 'kc1': '', 'kc3': '', 'setrgbf': '\\E[38:2:%p1%d:%p2%d:%p3%dm', 'setrgbb': '\\E[48:2:%p1%d:%p2%d:%p3%dm', 'Ss': '\\E[%p1%d\\sq', 'Se': '\\E[2\\sq', 'Cs': '\\E]12;%p1%s\\007', 'Cr': '\\E]112\\007', 'Setulc': '\\E[58:2:%p1%{65536}%/%d:%p1%{256}%/%{255}%&%d:%p1%{255}%&%d%;m', 'u6': '\\E[%i%d;%dR', 'u7': '\\E[6n', 'u8': '\\E[?%[;0123456789]c', 'u9': '\\E[c', 'PS': '\\E[200~', 'PE': '\\E[201~', 'BE': '\\E[?2004h', 'BD': '\\E[?2004l', 'XR': '\\E[>0q', 'Ms': '\\E]52;%p1%s;%p2%s\\E\\\\', 'RV': '\\E[>c', 'kxIN': '\\E[I', 'kxOUT': '\\E[O', 'fe': '\\E[?1004h', 'fd': '\\E[?1004l'}
string_capabilities.update({f'kf{n}': encode_keystring(modify_key_bytes(b'\x1b' + value, 0)) for (n, value) in zip(range(1, 13), b'OP OQ OR OS [15~ [17~ [18~ [19~ [20~ [21~ [23~ [24~'.split())})
string_capabilities.update({f'kf{offset + n}': encode_keystring(modify_key_bytes(b'\x1b' + value, mod)) for (offset, mod) in {12: 2, 24: 5, 36: 6, 48: 3, 60: 4}.items() for (n, value) in zip(range(1, 13), b'OP OQ [13~ OS [15~ [17~ [18~ [19~ [20~ [21~ [23~ [24~'.split()) if offset + n < 64})
string_capabilities.update({name.format(unmod=unmod, key=key): encode_keystring(modify_key_bytes(b'\x1b' + value, mod)) for (unmod, key, value) in zip('cuu1 cud1 cuf1 cub1 beg end home ich1 dch1 pp  np'.split(), 'UP   DN   RIT  LFT  BEG END HOM  IC   DC   PRV NXT'.split(), b'OA  OB   OC   OD   OE  OF  OH   [2~  [3~  [5~ [6~'.split()) for (name, mod) in {'k{unmod}': 0, 'k{key}': 2, 'k{key}3': 3, 'k{key}4': 4, 'k{key}5': 5, 'k{key}6': 6, 'k{key}7': 7}.items()})
termcap_aliases.update({'ac': 'acsc', 'bl': 'bel', 'md': 'bold', 'bt': 'cbt', 'kB': 'kcbt', 'cl': 'clear', 'vi': 'civis', 'vs': 'cvvis', 've': 'cnorm', 'cr': 'cr', 'cs': 'csr', 'LE': 'cub', 'le': 'cub1', 'DO': 'cud', 'do': 'cud1', 'UP': 'cuu', 'up': 'cuu1', 'nd': 'cuf1', 'RI': 'cuf', 'cm': 'cup', 'DC': 'dch', 'dc': 'dch1', 'mh': 'dim', 'DL': 'dl', 'dl': 'dl1', 'ec': 'ech', 'cd': 'ed', 'ce': 'el', 'cb': 'el1', 'vb': 'flash', 'ho': 'home', 'ch': 'hpa', 'ta': 'ht', 'st': 'hts', 'IC': 'ich', 'AL': 'il', 'al': 'il1', 'sf': 'ind', 'SF': 'indn', 'Ic': 'initc', 'oc': 'oc', 'kb': 'kbs', 'kl': 'kcub1', 'kd': 'kcud1', 'kr': 'kcuf1', 'ku': 'kcuu1', 'kh': 'khome', '@7': 'kend', 'kI': 'kich1', 'kD': 'kdch1', 'Km': 'kmous', 'kN': 'knp', 'kP': 'kpp', 'kR': 'kri', 'kF': 'kind', 'rc': 'rc', 'rp': 'rep', 'mr': 'rev', 'sr': 'ri', 'SR': 'rin', 'RA': 'rmam', 'te': 'rmcup', 'ei': 'rmir', 'se': 'rmso', 'ue': 'rmul', 'Te': 'rmxx', 'r1': 'rs1', 'sc': 'sc', 'AB': 'setab', 'AF': 'setaf', 'sa': 'sgr', 'me': 'sgr0', 'op': 'op', 'SA': 'smam', 'ti': 'smcup', 'im': 'smir', 'so': 'smso', 'us': 'smul', 'Ts': 'smxx', 'ct': 'tbc', 'cv': 'vpa', 'ZH': 'sitm', 'ZR': 'ritm', 'as': 'smacs', 'ae': 'rmacs', 'ks': 'smkx', 'ke': 'rmkx', '#2': 'kHOM', '#3': 'kIC', '#4': 'kLFT', '*4': 'kDC', '*7': 'kEND', '%c': 'kNXT', '%e': 'kPRV', '%i': 'kRIT', '%1': 'khlp', '&8': 'kund', 'K1': 'ka1', 'K3': 'ka3', 'K4': 'kc1', 'K5': 'kc3', 'ts': 'tsl', 'fs': 'fsl', 'ds': 'dsl', 'u6': 'u6', 'u7': 'u7', 'u8': 'u8', 'u9': 'u9'})
termcap_aliases.update({tc: f'kf{n}' for (n, tc) in enumerate('k1 k2 k3 k4 k5 k6 k7 k8 k9 k; F1 F2 F3 F4 F5 F6 F7 F8 F9 FA FB FC FD FE FF FG FH FI FJ FK FL FM FN FO FP FQ FR FS FT FU FV FW FX FY FZ Fa Fb Fc Fd Fe Ff Fg Fh Fi Fj Fk Fl Fm Fn Fo Fp Fq Fr'.split(), 1)})
queryable_capabilities = cast(Dict[str, str], numeric_capabilities.copy())
queryable_capabilities.update(string_capabilities)
extra = (bool_capabilities | numeric_capabilities.keys() | string_capabilities.keys()) - set(termcap_aliases.values())
no_termcap_for = frozenset('XR Ms RV kxIN kxOUT Cr Cs Se Ss Setulc Su Smulx Sync Tc PS PE BE BD setrgbf setrgbb fullkbd kUP kDN kbeg kBEG fe fd XF'.split() + [f'k{key}{mod}' for key in 'UP DN RIT LFT BEG END HOM IC DC PRV NXT'.split() for mod in range(3, 8)])
if extra - no_termcap_for:
    raise Exception(f'Termcap aliases not complete, missing: {extra - no_termcap_for}')
del extra

def generate_terminfo() -> str:
    if False:
        print('Hello World!')
    ans = ['|'.join(names)]
    ans.extend(sorted(bool_capabilities))
    ans.extend((f'{k}#{numeric_capabilities[k]}' for k in sorted(numeric_capabilities)))
    ans.extend((f'{k}={string_capabilities[k]}' for k in sorted(string_capabilities)))
    return ',\n\t'.join(ans) + ',\n'
octal_escape = re.compile('\\\\([0-7]{3})')
escape_escape = re.compile('\\\\[eE]')

def key_as_bytes(name: str) -> bytes:
    if False:
        i = 10
        return i + 15
    ans = string_capabilities[name]
    ans = octal_escape.sub(lambda m: chr(int(m.group(1), 8)), ans)
    ans = escape_escape.sub('\x1b', ans)
    return ans.encode('ascii')

def get_capabilities(query_string: str, opts: 'Options') -> Generator[str, None, None]:
    if False:
        i = 10
        return i + 15
    from .fast_data_types import ERROR_PREFIX

    def result(encoded_query_name: str, x: Optional[str]=None) -> str:
        if False:
            print('Hello World!')
        if x is None:
            return f'0+r{encoded_query_name}'
        return f"1+r{encoded_query_name}={hexlify(str(x).encode('utf-8')).decode('ascii')}"
    for encoded_query_name in query_string.split(';'):
        name = qname = unhexlify(encoded_query_name).decode('utf-8')
        if name in ('TN', 'name'):
            yield result(encoded_query_name, names[0])
        elif name.startswith('kitty-query-'):
            from kittens.query_terminal.main import get_result
            name = name[len('kitty-query-'):]
            rval = get_result(name)
            if rval is None:
                from .utils import log_error
                log_error('Unknown kitty terminfo query:', name)
                yield result(encoded_query_name)
            else:
                yield result(encoded_query_name, rval)
        else:
            try:
                val = queryable_capabilities[name]
            except KeyError:
                try:
                    qname = termcap_aliases[name]
                    val = queryable_capabilities[qname]
                except Exception:
                    from .utils import log_error
                    log_error(ERROR_PREFIX, 'Unknown terminfo property:', name)
                    yield result(encoded_query_name)
                    continue
            if qname in string_capabilities and '%' not in val:
                val = key_as_bytes(qname).decode('ascii')
            yield result(encoded_query_name, val)