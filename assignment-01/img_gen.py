"""Image Processing assignment from course SCC0251.

Assignment data:
    Title: "Assignment 1: image generation"
    Year: 2019 (Semester 01 / Fall)

Student data:
    Name: Felipe Alves Siqueira
    NoUSP: 9847706
    Undergraduation Student
"""
import typing as t
import inspect
import random

# import matplotlib.pyplot as plt
import numpy as np


class ImageGenerator:
    """Generate an image using a mathematical function."""

    def __init__(self,
                 img_dim: int,
                 Q: t.Union[int, float],
                 gen_func: t.Callable,
                 rand_seed: int = None):
        """Image Generator.

        Args:
            img_dim (int): dimension of the image to be generated. This
                means that the generated image will be square with dimen-
                sions (img_dim, img_dim).

            Q (float or int): numerical parameter to adjust given mathe-
                matical function.

            gen_func (Callable): mathematical function to generate each
                pixel of the image. Its signature must be:

                gen_func(x_coord, y_coord) or gen_func(x_coord, y_coord, Q)

            rand_seed (int, optional): random seed to be set. Used by random-
                based generator functions.
        """
        self.img_dim = img_dim
        self.Q = Q
        self.gen_func = gen_func
        self.rand_seed = rand_seed

        self.img = None

        try:
            self._gen_func_args = frozenset(
                inspect.getargs(self.gen_func.__code__)[0])

        except AttributeError:
            self._gen_func_args = frozenset()

    def generate(self) -> np.ndarray:
        """Generate an image based in fitted function and arguments."""
        if self.rand_seed is not None:
            random.seed(self.rand_seed)
            np.random.seed(self.rand_seed)

        extra_args = {}

        if "Q" in self._gen_func_args:
            extra_args["Q"] = self.Q

        if {"coord_x", "coord_y"}.issubset(self._gen_func_args):
            # Construct an image based on (x, y) coordinates, and
            # maybe Q's value
            self.img = np.fromfunction(
                self.gen_func, (self.img_dim, self.img_dim), **extra_args)
        else:
            # Construct an image not based on any coordinates (like
            # random construction), however can be based on Q's value
            self.img = np.array([
                self.gen_func(**extra_args) for _ in range(self.img_dim**2)
            ]).reshape((self.img_dim, self.img_dim))

        return self.img

    def rescale(self,
                img: t.Optional[np.ndarray] = None,
                val_max: t.Union[int, float] = 255,
                val_min: t.Union[int, float] = 0,
                cast_type: t.Optional[t.Any] = None) -> np.ndarray:
        """Rescale values of fitted image in [val_min, val_max] interval.

        Args:
            img (np.ndarray): image to rescale. If None is given, then
                the model will rescale the generated image.

            val_max (int or float): maximum value of the rescaled image.

            val_min (int or float): minimum value of the rescaled image.

            cast_type (type, optional): if given, cast the rescaled image
                to the given type.

        Returns:
            np.ndarray: rescaled image.

        Raises:
            TypeError: if calling this method before 'generate' method (which
                means that the instance attribute 'img' is NoneType)

            ValueError: is 'val_max' value is lesser than 'val_min' value.
        """
        update_fitted_img = False

        if img is None:
            img = self.img
            update_fitted_img = True

        if img is None:
            raise TypeError("Please call 'generate' method before rescaling.")

        if val_max < val_min:
            raise ValueError("'val_max' ({}) must be greater or equal "
                             "than 'val_min ({})".format(val_max, val_min))

        img = img.astype(float)

        img_min = img.min()
        img_max = img.max()

        if img_min == img_max:
            return img

        img = (
            val_min +
            (val_max - val_min) * np.divide(img - img_min, img_max - img_min))

        if cast_type:
            img = img.astype(cast_type)

        if update_fitted_img:
            self.img = img

        return img

    def downsampling(self, new_size: int) -> np.ndarray:
        """Reduces the fitted image dimension to 'new_size'."""
        if self.img is None:
            raise TypeError("Please call 'generate' method before rescaling.")

        img_dim, _ = self.img.shape

        stride = img_dim // new_size

        self.img = np.array(
            [[self.img[i * stride, j * stride] for j in range(new_size)]
             for i in range(new_size)])

        return self.img

    def quantisation(self, num_bits: int = 8) -> np.ndarray:
        """Rescale image to [0, 255] and shift its values by (8 - num_bits)."""
        if self.img is None:
            raise TypeError("Please call 'generate' method before rescaling.")

        if not isinstance(num_bits, int):
            raise TypeError("'num_bits' type must be integral")

        if not 0 < num_bits <= 8:
            raise ValueError("'num_bits' must be in-between 1 and 8.")

        self.rescale(val_max=255, val_min=0, cast_type=np.uint8)

        self.img >>= (8 - num_bits)

        return self.img

    def compare(self, ref_img: np.ndarray, plot: bool = False) -> float:
        """Compare generated image with reference image and return error.

        Args:
            ref_img (np.ndarray): image for reference

            plot(bool, optional): if true, plot images and its differences.

        Returns:
            float: value of RMSE (Root Mean Squared Error).
        """
        if self.img is None:
            raise TypeError("Please call 'generate' method before rescaling.")

        """
        # Run.codes doesn't seem to have 'matplotlib', but I decided to keep
        # this code because it's really useful for debugging.

        if plot:
            plt.subplot(131)
            plt.title("Generated (min={:.1f}, "
                      "max={:.1f})".format(self.img.min(), self.img.max()))
            plt.imshow(self.img, cmap="gray")

            plt.subplot(132)
            plt.title("Reference (min={:.1f}, "
                      "max={:.1f})".format(ref_img.min(), ref_img.max()))
            plt.imshow(ref_img, cmap="gray")

            plt.subplot(133)
            diff_img = self.img.astype(float) - ref_img.astype(float)

            diff_img = self.rescale(
                img=diff_img,
                val_max=1,
                val_min=0,
                cast_type=float,
            )

            plt.title("Difference (min={:.1f}, "
                      "max={:.1f})".format(diff_img.min(), diff_img.max()))
            plt.imshow(diff_img, cmap="gray", vmin=0, vmax=1)
            plt.colorbar()

            plt.show()
        """

        flatten_img_fit = self.img.astype(float).flatten()
        flatten_img_ref = ref_img.astype(float).flatten()

        sqr_error = np.power(flatten_img_fit - flatten_img_ref, 2.0).sum()

        return np.power(sqr_error, 0.5)


class RandomWalk(ImageGenerator):
    """Generate images using Random Walking strategy."""

    def __init__(self,
                 img_dim: int,
                 max_steps: int,
                 Q: t.Union[int, float],
                 start_point: t.Tuple[int, int] = (0, 0),
                 rand_seed: int = None):
        """Image Generator using random walkings.

        Args:
            Check 'ImageGenerator' class for more information.

            max_steps (int): number of steps (movements) of the random
                walking.

            start_point (tuple of ints, optional): a two-dimensional
                tuple which describes the initial point of the random-
                walking.
        """
        super().__init__(
            img_dim=img_dim,
            Q=Q,
            gen_func=None,
            rand_seed=rand_seed,
        )

        if not isinstance(max_steps, int) or not max_steps >= 0:
            raise ValueError("'max_steps' must be a natural number")

        if not start_point or len(start_point) < 2:
            raise ValueError("'start_point' must be a iterable with length 2.")

        self.max_steps = max_steps
        self.start_point = start_point

    def generate(self) -> np.ndarray:
        """Generate a image using random walking strategy."""
        if self.rand_seed is not None:
            random.seed(self.rand_seed)
            np.random.seed(self.rand_seed)

        coord_x, coord_y = self.start_point

        self.img = np.zeros((self.img_dim, self.img_dim))
        self.img[coord_y, coord_x] = 1

        for _ in range(self.max_steps):
            var_x, var_y = random.randint(-1, 1), random.randint(-1, 1)

            coord_x = (coord_x + var_x) % self.img_dim
            coord_y = (coord_y + var_y) % self.img_dim

            self.img[coord_y, coord_x] = 1

        return self.img


def main(subpath=None):
    def get_parameters():
        """Get parameters from assignment input file."""
        args = [str(input()).rstrip()] + [0] * 6

        for i in range(1, 7):
            args[i] = int(input())

        return args

    def gen_func_1(coord_x, coord_y):
        return coord_x * coord_y + 2.0 * coord_y

    def gen_func_2(coord_x, coord_y, Q):
        return np.abs(np.cos(coord_x / Q) + 2.0 * np.sin(coord_y / Q))

    def gen_func_3(coord_x, coord_y, Q):
        return np.abs(3.0 * (coord_x / Q) - np.cbrt(coord_y / Q))

    generator_funcs = {
        1: gen_func_1,
        2: gen_func_2,
        3: gen_func_3,
        4: random.random,
    }

    ref_img_filename, C, f_id, Q, N, B, S = get_parameters()

    if f_id != 5:
        img_gen = ImageGenerator(
            gen_func=generator_funcs.get(f_id),
            Q=Q,
            img_dim=C,
            rand_seed=S,
        )

    else:
        img_gen = RandomWalk(
            Q=Q,
            img_dim=C,
            rand_seed=S,
            start_point=(0, 0),
            max_steps=1 + C**2,
        )

    img_gen.generate()

    # Not really necessary, but this rescaling was asked by
    # the assignment documentation.
    img_gen.rescale(val_max=2**16-1, val_min=0)

    img_gen.downsampling(new_size=N)

    img_gen.quantisation(num_bits=B)

    if subpath:
        # Just used to help debugging.
        ref_img_filename = "/".join((subpath, ref_img_filename))

    ref_img = np.load(ref_img_filename)

    rmse = img_gen.compare(ref_img=ref_img, plot=False)

    print("{:.4f}".format(rmse))


if __name__ == "__main__":
    main(subpath=None)
