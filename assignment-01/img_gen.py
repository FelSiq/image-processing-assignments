"""Image Processing assignment from course SCC0251.

Assignment data:
    Title: "Assignment 1: image generation"
    Year: 2019 (Semester 01 / Fall)

Student data:
    Name: Felipe Alves Siqueira
    NoUSP: 9847706
    Undergraduation Student
"""
import matplotlib.pyplot as plt
import numpy as np
import imageio
import random
import sys


GENERATOR_FUNC = {
    1: lambda x, y, _: x*y + 2*y,
    2: lambda x, y, Q: abs(np.cos(x/Q) + 2*np.sin(y/Q)),
    3: lambda x, y, Q: abs(3*(x/Q) - (y/Q)**1/3),
    4: lambda x, y, _: random.random(),
}
"""."""


def get_parameters():
    """Something.

    Args:

    Returns:

    Raises:
    """
    args = [str(input()).strip()] + [0] * 6
    
    for i in range(1, 7):
        args[i] = int(input())

    return args


def RMSE(x, y):
    """Calculate the Root Mean Squared Error from two arrays."""
    return ((x.flatten() - y.flatten())**2.0).sum()**0.5


def get_gen_func(f_id):
    """Something.

    Args:

    Returns:

    Raises:
    """
    return GENERATOR_FUNC.get(f_id)


def rescale_image(img, max_val=2**16, min_val=0):
    """."""
    img_min = img.min()
    img_max = img.max()

    if img_min == img_max:
        return img

    return ((max_val - min_val) * (img - img_min)
            / (img_max - img_min) + min_val)


def randomwalk(C, start_pos=(0, 0), max_steps=100):
    """."""
    x, y = start_pos
    img = np.zeros((C, C))
    img[y, x] = 1
    for _ in range(max_steps):
        dx = 2 * random.random() - 1
        dy = 2 * random.random() - 1
    
        x = round((x + dx)) % C
        y = round((y + dy)) % C
        img[y, x] = 1

    return rescale_image(img)


def generate_image(f_id, C, Q, B):
    """Something.

    Args:

    Returns:

    Raises:
    """
    if f_id == 5:
        return randomwalk(C=C, max_steps=1 + C**2)

    gen_func = get_gen_func(f_id)

    if not gen_func:
        raise ValueError('Invalid generator function id ({})'.format(f_id))

    gen_img = np.array([
        [gen_func(x, y, Q)
        for y in range(C)]
        for x in range(C)
    ])

    return rescale_image(gen_img)

def main():
    filename, C, f_id, Q, N, B, S = get_parameters()

    random.seed(S)

    gen_img = generate_image(f_id, C, Q, B)

    plt.imshow(gen_img, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
