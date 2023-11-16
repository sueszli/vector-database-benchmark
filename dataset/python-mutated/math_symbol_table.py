import re
from docutils.parsers.rst import Directive
from matplotlib import _mathtext, _mathtext_data
bb_pattern = re.compile('Bbb[A-Z]')
scr_pattern = re.compile('scr[a-zA-Z]')
frak_pattern = re.compile('frak[A-Z]')
symbols = [['Lower-case Greek', 4, ('\\alpha', '\\beta', '\\gamma', '\\chi', '\\delta', '\\epsilon', '\\eta', '\\iota', '\\kappa', '\\lambda', '\\mu', '\\nu', '\\omega', '\\phi', '\\pi', '\\psi', '\\rho', '\\sigma', '\\tau', '\\theta', '\\upsilon', '\\xi', '\\zeta', '\\digamma', '\\varepsilon', '\\varkappa', '\\varphi', '\\varpi', '\\varrho', '\\varsigma', '\\vartheta')], ['Upper-case Greek', 4, ('\\Delta', '\\Gamma', '\\Lambda', '\\Omega', '\\Phi', '\\Pi', '\\Psi', '\\Sigma', '\\Theta', '\\Upsilon', '\\Xi')], ['Hebrew', 6, ('\\aleph', '\\beth', '\\gimel', '\\daleth')], ['Latin named characters', 6, '\\aa \\AA \\ae \\AE \\oe \\OE \\O \\o \\thorn \\Thorn \\ss \\eth \\dh \\DH'.split()], ['Delimiters', 5, _mathtext.Parser._delims], ['Big symbols', 5, _mathtext.Parser._overunder_symbols | _mathtext.Parser._dropsub_symbols], ['Standard function names', 5, {f'\\{fn}' for fn in _mathtext.Parser._function_names}], ['Binary operation symbols', 4, _mathtext.Parser._binary_operators], ['Relation symbols', 4, _mathtext.Parser._relation_symbols], ['Arrow symbols', 4, _mathtext.Parser._arrow_symbols], ['Dot symbols', 4, '\\cdots \\vdots \\ldots \\ddots \\adots \\Colon \\therefore \\because'.split()], ['Black-board characters', 6, [f'\\{symbol}' for symbol in _mathtext_data.tex2uni if re.match(bb_pattern, symbol)]], ['Script characters', 6, [f'\\{symbol}' for symbol in _mathtext_data.tex2uni if re.match(scr_pattern, symbol)]], ['Fraktur characters', 6, [f'\\{symbol}' for symbol in _mathtext_data.tex2uni if re.match(frak_pattern, symbol)]], ['Miscellaneous symbols', 4, '\\neg \\infty \\forall \\wp \\exists \\bigstar \\angle \\partial\n     \\nexists \\measuredangle \\emptyset \\sphericalangle \\clubsuit\n     \\varnothing \\complement \\diamondsuit \\imath \\Finv \\triangledown\n     \\heartsuit \\jmath \\Game \\spadesuit \\ell \\hbar \\vartriangle\n     \\hslash \\blacksquare \\blacktriangle \\sharp \\increment\n     \\prime \\blacktriangledown \\Im \\flat \\backprime \\Re \\natural\n     \\circledS \\P \\copyright \\circledR \\S \\yen \\checkmark \\$\n     \\cent \\triangle \\QED \\sinewave \\dag \\ddag \\perthousand \\ac\n     \\lambdabar \\L \\l \\degree \\danger \\maltese \\clubsuitopen\n     \\i \\hermitmatrix \\sterling \\nabla \\mho'.split()]]

def run(state_machine):
    if False:
        for i in range(10):
            print('nop')

    def render_symbol(sym, ignore_variant=False):
        if False:
            i = 10
            return i + 15
        if ignore_variant and sym not in ('\\varnothing', '\\varlrtriangle'):
            sym = sym.replace('\\var', '\\')
        if sym.startswith('\\'):
            sym = sym.lstrip('\\')
            if sym not in _mathtext.Parser._overunder_functions | _mathtext.Parser._function_names:
                sym = chr(_mathtext_data.tex2uni[sym])
        return f'\\{sym}' if sym in ('\\', '|', '+', '-', '*') else sym
    lines = []
    for (category, columns, syms) in symbols:
        syms = sorted(syms, key=lambda sym: (render_symbol(sym, ignore_variant=True), sym.startswith('\\var')), reverse=category == 'Hebrew')
        rendered_syms = [f'{render_symbol(sym)} ``{sym}``' for sym in syms]
        columns = min(columns, len(syms))
        lines.append('**%s**' % category)
        lines.append('')
        max_width = max(map(len, rendered_syms))
        header = ('=' * max_width + ' ') * columns
        lines.append(header.rstrip())
        for part in range(0, len(rendered_syms), columns):
            row = ' '.join((sym.rjust(max_width) for sym in rendered_syms[part:part + columns]))
            lines.append(row)
        lines.append(header.rstrip())
        lines.append('')
    state_machine.insert_input(lines, 'Symbol table')
    return []

class MathSymbolTableDirective(Directive):
    has_content = False
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {}

    def run(self):
        if False:
            print('Hello World!')
        return run(self.state_machine)

def setup(app):
    if False:
        i = 10
        return i + 15
    app.add_directive('math_symbol_table', MathSymbolTableDirective)
    metadata = {'parallel_read_safe': True, 'parallel_write_safe': True}
    return metadata
if __name__ == '__main__':
    print('SYMBOLS NOT IN STIX:')
    all_symbols = {}
    for (category, columns, syms) in symbols:
        if category == 'Standard Function Names':
            continue
        for sym in syms:
            if len(sym) > 1:
                all_symbols[sym[1:]] = None
                if sym[1:] not in _mathtext_data.tex2uni:
                    print(sym)
    all_symbols.update({v[1:]: k for (k, v) in _mathtext.Parser._accent_map.items()})
    all_symbols.update({v: v for v in _mathtext.Parser._wide_accents})
    print('SYMBOLS NOT IN TABLE:')
    for (sym, val) in _mathtext_data.tex2uni.items():
        if sym not in all_symbols:
            print(f'{sym} = {chr(val)}')