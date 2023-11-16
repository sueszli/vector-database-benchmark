"""Find Nash equilibria for constant- or general-sum 2-player games.

Non-matrix games are handled by computing the normal (bimatrix) form.
The algorithms used are:
* direct computation of pure equilibria.
* linear programming to find equilibria for constant-sum games.
* iterated dominance to reduce the action space.
* reverse search vertex enumeration (if using lrsnash) to find all general-sum
  equilibria.
* support enumeration (if using nashpy) to find all general-sum equilibria.
* Lemke-Howson enumeration (if using nashpy) to find one general-sum
  equilibrium.
The general-sum mixed-equilibrium algorithms are likely to work well for tens of
actions, but less likely to scale beyond that.
"""
import fractions
import os
import subprocess
import tempfile
import warnings
import nashpy
import numpy as np

@np.vectorize
def to_fraction_str(x, lrsnash_max_denom):
    if False:
        return 10
    return str(fractions.Fraction(x).limit_denominator(lrsnash_max_denom))

def lrs_solve(row_payoffs, col_payoffs, lrsnash_max_denom, lrsnash_path):
    if False:
        return 10
    'Find all Nash equilibria using the lrsnash solver.\n\n  `lrsnash` uses reverse search vertex enumeration on rational polytopes.\n  For more info, see: http://cgm.cs.mcgill.ca/~avis/C/lrslib/USERGUIDE.html#nash\n\n  Args:\n    row_payoffs: payoffs for row player\n    col_payoffs: payoffs for column player\n    lrsnash_max_denom: maximum denominator\n    lrsnash_path: path for temporary files\n\n  Yields:\n    (row_mixture, col_mixture), numpy vectors of float64s.\n  '
    (num_rows, num_cols) = row_payoffs.shape
    (game_file, game_file_path) = tempfile.mkstemp()
    try:
        game_file = os.fdopen(game_file, 'w')
        game_file.write('%d %d\n\n' % (num_rows, num_cols))
        for row in range(num_rows):
            game_file.write(' '.join(to_fraction_str(row_payoffs[row], lrsnash_max_denom)) + '\n')
        game_file.write('\n')
        for row in range(num_rows):
            game_file.write(' '.join(to_fraction_str(col_payoffs[row], lrsnash_max_denom)) + '\n')
        game_file.write('\n')
        game_file.close()
        lrs = subprocess.Popen([lrsnash_path or 'lrsnash', '-s', game_file_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        col_mixtures = []
        for line in lrs.stdout:
            if len(line) <= 1 or line[:1] == b'*':
                continue
            line = np.asfarray([fractions.Fraction(x) for x in line.decode().split()])
            if line[0] == 2:
                col_mixtures.append(line[1:-1])
            else:
                row_mixture = line[1:-1]
                for col_mixture in col_mixtures:
                    yield (row_mixture, col_mixture)
                col_mixtures = []
    finally:
        os.remove(game_file_path)

def lemke_howson_solve(row_payoffs, col_payoffs):
    if False:
        i = 10
        return i + 15
    'Find Nash equilibria using the Lemke-Howson algorithm.\n\n  The algorithm is not guaranteed to find all equilibria. Also it can yield\n  wrong answers if the game is degenerate (but raises warnings in that case).\n  Args:\n    row_payoffs: payoffs for row player\n    col_payoffs: payoffs for column player\n  Yields:\n    (row_mixture, col_mixture), numpy vectors of float64s.\n  '
    showwarning = warnings.showwarning
    warned_degenerate = [False]

    def showwarning_check_degenerate(message, *args, **kwargs):
        if False:
            return 10
        if 'Your game could be degenerate.' in str(message):
            warned_degenerate[0] = True
        showwarning(message, *args, **kwargs)
    try:
        warnings.showwarning = showwarning_check_degenerate
        for (row_mixture, col_mixture) in nashpy.Game(row_payoffs, col_payoffs).lemke_howson_enumeration():
            if warned_degenerate[0]:
                if row_mixture.shape != row_payoffs.shape[:1] or col_mixture.shape != row_payoffs.shape[1:]:
                    warnings.warn('Discarding ill-shaped solution.')
                    continue
                if not np.isfinite(row_mixture).all() or not np.isfinite(col_mixture).all():
                    warnings.warn('Discarding non-finite solution.')
                    continue
            yield (row_mixture, col_mixture)
    finally:
        warnings.showwarning = showwarning