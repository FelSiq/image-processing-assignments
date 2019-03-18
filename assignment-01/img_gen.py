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
import random


GENERATOR_FUNC = {
    1: lambda x, y, _: x*y + 2.0*y,
    2: lambda x, y, Q: abs(np.cos(x/Q) + 2*np.sin(y/Q)),
    3: lambda x, y, Q: abs(3*(x/Q) - np.power(y/Q, 1/3)),
    4: lambda x, y, _: np.array([random.uniform(0, 1) for _ in range(len(x))]),
}
"""."""


def get_parameters():
    """Something.

    Args:

    Returns:

    Raises:
    """
    args = [str(input()).rstrip()] + [0] * 6
    
    for i in range(1, 7):
        args[i] = int(input())

    print(args)

    return args


def RMSE(x, y):
    """Calculate the Root Mean Squared Error from two arrays."""
    x = x.astype(float)
    y = y.astype(float)
    return ((x.flatten() - y.flatten())**2.0).sum()**0.5


def get_gen_func(f_id):
    """Something.

    Args:

    Returns:

    Raises:
    """
    return GENERATOR_FUNC.get(f_id)


def rescale_img(img, max_val=2**16-1, min_val=0, dtype=np.uint16):
    """Something.

    Args:

    Returns:

    Raises:
    """
    img_min = img.min()
    img_max = img.max()

    if img_min == img_max:
        return img

    img = img.astype(float)

    return ((max_val - min_val) * (img - img_min)
            / (img_max - img_min) + min_val).astype(dtype)


def randomwalk(C, S, start_pos=(0, 0), max_steps=100):
    """Something.

    Args:

    Returns:

    Raises:
    """
    x, y = start_pos
    img = np.zeros((C, C))
    img[y, x] = 1

    random.seed(S)
    np.random.seed(S)

    for _ in range(max_steps):
        dx = random.uniform(-1, 1)
        dy = random.uniform(-1, 1)

        x = round(x + dx) % C
        y = round(y + dy) % C
        img[y, x] = 1

    return rescale_img(img)


def generate_img(f_id, C, Q, S):
    """Something.

    Args:

    Returns:

    Raises:
    """
    if f_id == 5:
        return randomwalk(C=C, max_steps=1 + C**2, S=S)

    gen_func = get_gen_func(f_id)

    if not gen_func:
        raise ValueError('Invalid generator function id ({})'.format(f_id))

    random.seed(S)
    gen_img = np.array([
        gen_func(np.repeat(x, C), np.arange(C), Q)
        for x in range(C)
    ])

    return rescale_img(gen_img)

def img_downsampling(img, N):
    """Something.

    Args:

    Returns:

    Raises:
    """
    stride = img.shape[0] // N

    img_downsampled = np.array([
      [img[j * stride, i * stride]
      for i in range(N)]
      for j in range(N)
    ])

    return img_downsampled

def img_quantisation(img, B):
    """Something.

    Args:

    Returns:

    Raises:
    """
    img = rescale_img(
        img,
        max_val=2**8-1,
        min_val=0,
        dtype=np.uint8)

    img >>= (8-B)

    return img

def main(subpath=None, plot=False):
    """Something.

    Args:

    Returns:

    Raises:
    """
    filename, C, f_id, Q, N, B, S = get_parameters()

    gen_img = generate_img(f_id, C, Q, S)

    gen_img = img_downsampling(gen_img, N) 

    gen_img = img_quantisation(gen_img, B)

    if subpath:
        filename = "/".join((subpath, filename))

    ref_img = np.load(filename).astype(np.uint8)

    if plot:
        plt.subplot(131)
        plt.title("Generated (min={:.4f}"
                  "max={:.4f})".format(gen_img.min(), gen_img.max()))
        plt.imshow(gen_img, cmap="gray")

        plt.subplot(132)
        plt.title("Reference (min={:.4f}"
                  "max={:.4f})".format(ref_img.min(), ref_img.max()))
        plt.imshow(ref_img, cmap="gray")

        plt.subplot(133)
        plt.title("Difference")
        plt.imshow(abs(ref_img.astype(float) - gen_img.astype(float)), cmap="gray")

        plt.show()

    rmse = RMSE(gen_img, ref_img)

    print("{:.4f}".format(rmse))


if __name__ == "__main__":
    main(subpath="./documents", plot=True)
