import numpy as np
import matplotlib.pyplot as plt

def r(t):
    return t * (t >= 0)

def rect(t):
    return np.where(np.logical_and(t >= 0, t <= 2), 1, 0)

t = np.linspace(-4, 4, 1000)
signal = r(t) * rect((t - 1) / 2)

plt.plot(t, signal)
plt.xlabel('t')
plt.ylabel('r(t) * rect((t-1)/2)')
plt.title('Signal: r(t) * rect((t-1)/2)')
plt.grid()
plt.show()
