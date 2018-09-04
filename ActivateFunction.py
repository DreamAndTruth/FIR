import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

x = np.linspace(-10.0, 10.0, 1000)

f1 = 1 / (1 + np.exp(-x))
f2 = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
f3 = x / (1 + np.abs(x))

fig = plt.figure(1)
f1, = plt.plot(x, f1)
f2, = plt.plot(x, f2)
f3, = plt.plot(x, f3)
plt.xlabel('x')
plt.ylabel('Activate Value')
plt.title('Activate Function')
plt.grid(True)
plt.legend([f1, f2, f3], ['f1', 'f2', 'f3'])
plt.show()