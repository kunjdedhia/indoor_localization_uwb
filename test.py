import numpy as np
import matplotlib.pyplot as plt

plt.scatter(np.asarray(1, 2, 3, 4), np.asarray(1, 2, 3, 4), s = 2)
plt.xlabel('R 109 at 1.016m')
plt.ylabel('R 108 at 2.032m')
plt.savefig('test.png')