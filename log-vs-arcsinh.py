import numpy as np
import matplotlib.pyplot as plt

x_max = 5
x_min = -5
x = np.arange(x_min, x_max, 1e-3)
x_limited = np.arange(1e-5, x_max, 1e-3)
y_arcsinh = np.arcsinh(x/2.)
y_log = np.log(x_limited)
y_linear = x

fig = plt.figure(figsize= (5,5))
plt.plot(x, y_arcsinh)
plt.plot(x_limited, y_log)
plt.plot(x, y_linear)
plt.axvline(x=0, c="black", lw=1, ls="--")
plt.axhline(y=0, c="black", lw=1, ls="--")
plt.axis("equal")
plt.xlim([x_min, x_max])
plt.ylim([x_min, x_max])
plt.savefig("log-vs-arcsinh.png", dpi=200, bbox_inches="tight")
plt.close()
