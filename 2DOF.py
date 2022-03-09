from matplotlib import cm
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
            return np.array(1000, 1000)
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


def calculate_arm_pos(final_pos, target, radius=1):
    """
    Gernerates a plot
    
    final_pos: final position of the gripper
    target: thing we are tryna grab
    radius: circle of inversion

    returns (normal distance, transformed distance)
    """
    trans_pos = transform(final_pos, radius)
    target = np.array(target)
    trans_target = np.array(transform(target, radius))
    norm_d = np.array(final_pos) - target
    trans_d = np.array(trans_pos) - trans_target

    return [np.linalg.norm(norm_d), np.linalg.norm(trans_d)]


def go_brr(arm_length=[1, 1], target=[2, 0]):
    first_rot = np.linspace(0, 2 * np.pi, 50)
    second_rot = np.linspace(0, 2 * np.pi, 50)

    rst = np.array([0, 0, 0, 0])
    for r1 in first_rot:
        for r2 in second_rot:
            end = [
                arm_length[0] * np.cos(r1) + arm_length[1] * np.cos(r2),
                arm_length[0] * np.sin(r1) + arm_length[1] * np.sin(r2)
            ]
            bruh = calculate_arm_pos(end, target)
            proto_row = [r1, r2, bruh[0], bruh[1]]
            rst = np.vstack((rst, proto_row))

    rst = rst[1:]
    return rst.T


fig = plt.figure(figsize=plt.figaspect(0.5))

#===============
#  First subplot
#===============
# set up the axes for the first plot
ax1 = fig.add_subplot(1, 2, 1, projection='3d')

#===============
# Second subplot
#===============
# set up the axes for the second plot
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

ax1.set_zlim(0, 5)
ax2.set_zlim(0, 5)
if __name__ == "__main__":
    mtx = go_brr(arm_length=[1, 1], target=[2, 0])
    ax1.plot_trisurf(mtx[0], mtx[1], mtx[2])
    ax1.set_title("Normal Euclidean Distance, Arm Length (1,1), Target (2, 0)")

    ax2.plot_trisurf(mtx[0], mtx[1], mtx[3])
    ax2.set_title(
        "Transformed Euclidean Distance, Arm Length (1,1), Target (2, 0)")

    print(mtx[0][539], mtx[1][539])
    plt.show()
