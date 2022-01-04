import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import trapz
import math
plt.style.use('seaborn-pastel')


def Curve_integral_animation(fu, x_limits, y_limits, a, b, N):
    fig = plt.figure()
    ax = plt.axes(xlim=(x_limits[0], x_limits[1]),
                  ylim=(y_limits[0], y_limits[1]))
    line, = ax.plot([], [], lw=1)
    pts = [[0, 0], [0, 0], [0, 0], [0, 0]]
    patch = plt.Polygon(pts)
    ax.add_patch(patch)
    ax.set_title(f'I(f) in interval ({a},{b})')
    title = ax.text(0.5, 0.85, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                    transform=ax.transAxes, ha="center")

    def animate(i):
        x = np.linspace(a, a + (i * (b - a) / N), i + 1)
        y = fu(x)
        title.set_text(
            f"x={'%.2f' % x[i]} \n I(f)={'%.2f' % trapz(y,x)}")
        func_points = [list(j) for j in zip(x, y)]
        patch.set_xy([[x[i], 0]] + [[x[0], 0]] + func_points)
        line.set_data(x, y)
        return line, patch, title

    anim = FuncAnimation(fig, animate,
                         frames=N + 1, interval=1, blit=True, repeat=False)

    plt.show()


def f(r):
    return 5 * np.exp(-5 * r)


Curve_integral_animation(f, [1, 2], [-0.1, 0.1],
                         0, 3, 100)
