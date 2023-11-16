import matplotlib.pyplot as plt
import numpy as np
from mlxtend.plotting import checkerboard_plot
plt.switch_backend('agg')

def test_runs():
    if False:
        for i in range(10):
            print('nop')
    ary = np.random.random((6, 4))
    checkerboard_plot(ary, col_labels=['abc', 'def', 'ghi', 'jkl'], row_labels=['sample %d' % i for i in range(1, 6)], cell_colors=['skyblue', 'whitesmoke'], font_colors=['black', 'black'], figsize=(5, 5))