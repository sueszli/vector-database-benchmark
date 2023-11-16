"""
============
Findobj Demo
============

Recursively find all objects that match some criteria
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.text as text
a = np.arange(0, 3, 0.02)
b = np.arange(0, 3, 0.02)
c = np.exp(a)
d = c[::-1]
(fig, ax) = plt.subplots()
plt.plot(a, c, 'k--', a, d, 'k:', a, c + d, 'k')
plt.legend(('Model length', 'Data length', 'Total message length'), loc='upper center', shadow=True)
plt.ylim([-1, 20])
plt.grid(False)
plt.xlabel('Model complexity --->')
plt.ylabel('Message length --->')
plt.title('Minimum Message Length')

def myfunc(x):
    if False:
        for i in range(10):
            print('nop')
    return hasattr(x, 'set_color') and (not hasattr(x, 'set_facecolor'))
for o in fig.findobj(myfunc):
    o.set_color('blue')
for o in fig.findobj(text.Text):
    o.set_fontstyle('italic')
plt.show()