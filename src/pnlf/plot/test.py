from style import figsize, newfig

import numpy as np 
import matplotlib.pyplot as plt 


x = np.linspace(0,6)
y = np.sin(x)

plt.plot(x,y)
plt.savefig('test.pgf')
plt.savefig('test.pdf')