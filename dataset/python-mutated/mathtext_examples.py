"""
=================
Mathtext Examples
=================

Selected features of Matplotlib's math rendering engine.
"""
import re
import subprocess
import sys
import matplotlib.pyplot as plt
mathtext_demos = {'Header demo': '$W^{3\\beta}_{\\delta_1 \\rho_1 \\sigma_2} = U^{3\\beta}_{\\delta_1 \\rho_1} + \\frac{1}{8 \\pi 2} \\int^{\\alpha_2}_{\\alpha_2} d \\alpha^\\prime_2 \\left[\\frac{ U^{2\\beta}_{\\delta_1 \\rho_1} - \\alpha^\\prime_2U^{1\\beta}_{\\rho_1 \\sigma_2} }{U^{0\\beta}_{\\rho_1 \\sigma_2}}\\right]$', 'Subscripts and superscripts': '$\\alpha_i > \\beta_i,\\ \\alpha_{i+1}^j = {\\rm sin}(2\\pi f_j t_i) e^{-5 t_i/\\tau},\\ \\ldots$', 'Fractions, binomials and stacked numbers': '$\\frac{3}{4},\\ \\binom{3}{4},\\ \\genfrac{}{}{0}{}{3}{4},\\ \\left(\\frac{5 - \\frac{1}{x}}{4}\\right),\\ \\ldots$', 'Radicals': '$\\sqrt{2},\\ \\sqrt[3]{x},\\ \\ldots$', 'Fonts': '$\\mathrm{Roman}\\ , \\ \\mathit{Italic}\\ , \\ \\mathtt{Typewriter} \\ \\mathrm{or}\\ \\mathcal{CALLIGRAPHY}$', 'Accents': '$\\acute a,\\ \\bar a,\\ \\breve a,\\ \\dot a,\\ \\ddot a, \\ \\grave a, \\ \\hat a,\\ \\tilde a,\\ \\vec a,\\ \\widehat{xyz},\\ \\widetilde{xyz},\\ \\ldots$', 'Greek, Hebrew': '$\\alpha,\\ \\beta,\\ \\chi,\\ \\delta,\\ \\lambda,\\ \\mu,\\ \\Delta,\\ \\Gamma,\\ \\Omega,\\ \\Phi,\\ \\Pi,\\ \\Upsilon,\\ \\nabla,\\ \\aleph,\\ \\beth,\\ \\daleth,\\ \\gimel,\\ \\ldots$', 'Delimiters, functions and Symbols': '$\\coprod,\\ \\int,\\ \\oint,\\ \\prod,\\ \\sum,\\ \\log,\\ \\sin,\\ \\approx,\\ \\oplus,\\ \\star,\\ \\varpropto,\\ \\infty,\\ \\partial,\\ \\Re,\\ \\leftrightsquigarrow, \\ \\ldots$'}
n_lines = len(mathtext_demos)

def doall():
    if False:
        i = 10
        return i + 15
    mpl_grey_rgb = (51 / 255, 51 / 255, 51 / 255)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.9], facecolor='white', frameon=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Matplotlib's math rendering engine", color=mpl_grey_rgb, fontsize=14, weight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    line_axesfrac = 1 / n_lines
    full_demo = mathtext_demos['Header demo']
    ax.annotate(full_demo, xy=(0.5, 1.0 - 0.59 * line_axesfrac), color='tab:orange', ha='center', fontsize=20)
    for (i_line, (title, demo)) in enumerate(mathtext_demos.items()):
        print(i_line, demo)
        if i_line == 0:
            continue
        baseline = 1 - i_line * line_axesfrac
        baseline_next = baseline - line_axesfrac
        fill_color = ['white', 'tab:blue'][i_line % 2]
        ax.axhspan(baseline, baseline_next, color=fill_color, alpha=0.2)
        ax.annotate(f'{title}:', xy=(0.06, baseline - 0.3 * line_axesfrac), color=mpl_grey_rgb, weight='bold')
        ax.annotate(demo, xy=(0.04, baseline - 0.75 * line_axesfrac), color=mpl_grey_rgb, fontsize=16)
    plt.show()
if '--latex' in sys.argv:
    with open('mathtext_examples.ltx', 'w') as fd:
        fd.write('\\documentclass{article}\n')
        fd.write('\\usepackage{amsmath, amssymb}\n')
        fd.write('\\begin{document}\n')
        fd.write('\\begin{enumerate}\n')
        for s in mathtext_demos.values():
            s = re.sub('(?<!\\\\)\\$', '$$', s)
            fd.write('\\item %s\n' % s)
        fd.write('\\end{enumerate}\n')
        fd.write('\\end{document}\n')
    subprocess.call(['pdflatex', 'mathtext_examples.ltx'])
else:
    doall()