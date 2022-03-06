from matplotlib.animation import FuncAnimation
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt


def transform(point, radius):
    """
    Transforms a POINT (regular array) with inversion
    with respect to a circle with radius RADIUS, centered
    at the origin
    """
    distance = np.linalg.norm(point)
    if distance < radius:

        if distance == 0:
            return point
        major = radius * radius / distance

        pt = np.array(point) * major / distance
        return pt

    leg = sqrt(abs(distance * distance - radius * radius))

    x1 = 0
    y1 = 0
    r1 = radius
    x2 = point[0]
    y2 = point[1]
    r2 = leg

    dx, dy = x2 - x1, y2 - y1
    d = sqrt(dx * dx + dy * dy)

    a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    h = sqrt(r1 * r1 - a * a)
    xm = x1 + a * dx / d
    ym = y1 + a * dy / d
    xs1 = xm + h * dy / d
    xs2 = xm - h * dy / d
    ys1 = ym - h * dx / d
    ys2 = ym + h * dx / d

    return [(xs1 + xs2) / 2, (ys1 + ys2) / 2]


def generate_plot(arr):

    theta = np.linspace(0, 2 * np.pi, 100)

    # Circle of inversion
    r = np.linalg.norm(np.array(arr[0]))
    x1 = r * np.cos(theta)
    x2 = r * np.sin(theta)

    v = [[0, 0]]
    for vector in arr:
        if len(v) == 1:
            v = np.linspace(v[-1], vector, 100)
            ax[1].plot(0, 0)
            ax[0].plot(v.T[0], v.T[1])
            continue
        v = np.linspace(v[-1], vector, 100)
        ax[0].plot(v.T[0], v.T[1])

        arr = [[], []]
        for j in range(100):
            rst = transform([v[j][0], v[j][1]], r)
            arr[0].append(rst[0])
            arr[1].append(rst[1])

        ax[1].plot(arr[0], arr[1])

    ax[1].plot(x1, x2)
    ax[1].set_aspect(1)
    ax[0].plot(x1, x2)
    ax[0].set_aspect(1)
    ax[0].set_xlim(-5, 5)
    ax[0].set_ylim(-5, 5)
    ax[1].set_xlim(-2, 2)
    ax[1].set_ylim(-2, 2)

    return (ax[0].plot(), ax[1].plot())


def init():
    ax[0].set_xlim(-5, 5)
    ax[0].set_ylim(-5, 5)
    ax[1].set_xlim(-5, 5)
    ax[1].set_ylim(-5, 5)
    return plt.plot()[0]


def animate(frame, r1=1, r2=1.5, r3=2.5, r4=1):
    """
    Spins the first wheel 2pi radians, second 4pi, third, 8pi
    """

    first = np.array([r1 * np.cos(frame), r1 * np.sin(frame)])
    second = first + np.array([r2 * np.cos(frame * 2), r2 * np.sin(frame * 2)])
    third = second + np.array([r3 * np.cos(frame * 4), r3 * np.sin(frame * 4)])
    fourth = third + np.array([r4 * np.cos(frame * 8), r4 * np.sin(frame * 8)])
    ax[0].clear()
    ax[1].clear()
    return generate_plot([first, second, third, fourth])[0]


fig, ax = plt.subplots(1, 2)

if __name__ == "__main__":
    ani = FuncAnimation(
        fig,
        animate,
        frames=np.linspace(0, 2 * np.pi, 1000),
        blit=True,
    )
    ani.save("4DOF.gif", fps=60)
    plt.show()
