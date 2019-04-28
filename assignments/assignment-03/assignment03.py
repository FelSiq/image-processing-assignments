"""Image Processing assignment from course SCC0251.

Assignment data:
    Title: "Assignment 2: image restoration"
    Year: 2019 (Semester 01 / Fall)

Student data:
    Name: Felipe Alves Siqueira
    NoUSP: 9847706
    Undergraduation Student
"""
import typing as t

import numpy as np
import scipy
import imageio
import matplotlib.pyplot as plt


class ImageRestoration:
    def __init__(self,
                 img_reference: np.ndarray,
                 img_degraded: np.ndarray,
                 filter_type: int,
                 gamma: float,
                 size: int,
                 mode: t.Optional[str] = None,
                 sigma: t.Optional[float] = None):
        """."""
        if filter_type not in (1, 2):
            raise ValueError("Unknown filter type '{}'. "
                             "Expecting 1 or 2.".format(filter_type))

        if mode not in ("average", "robust"):
            raise ValueError("Unknown mode parameter '{}'. Expecting "
                             "'average' or 'robust'".format(mode))

        if sigma <= 0:
            raise ValueError("sigma must be a positive number. Got "
                             "{}".format(sigma))

        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1] interval. Got "
                             "{}".format(gamma))

        if not (mode is None ^ sigma is None):
            raise TypeError("'mode' and 'sigma' can't be both None or "
                            "both given. Choose precisely one.")

        self.img_ref = self._open_img(img_reference)
        self.img_deg = self._open_img(img_degraded)

        self.img_restored = None

        self.gamma = gamma
        self.size = size
        self.filter_type = filter_type
        self.sigma = sigma
        self.mode = mode

    def _open_img(self, filepath: str) -> np.ndarray:
        """Load the image from the given filepath."""
        return imageio.imgread(filepath)

    def adaptive_denoising(self) -> np.ndarray:
        """."""

    def constrained_lsf(self) -> np.ndarray:
        """."""

    def restore(self) -> np.ndarray:
        if self.img_deg is None:
            raise TypeError("Please load a degenerated image "
                            "before restorating.")
        """."""

    def compare(self) -> float:
        """Return RMSE between the restored and reference images."""
        if self.img_ref is None or self.img_restored is None:
            raise TypeError("Please load a reference image and call "
                            "'restore' method before comparing.")

        rmse = np.power(((self.img_restored - self.img_ref)**2.0).sum(), 0.5)
        return np.divide(rmse, np.multiply(self.img_restored.shape))

    def plot(self) -> None:
        """."""


if __name__ == "__main__":
    def get_parameters() -> t.Dict[str, t.Any]:
        """Get test case parameters."""
        param_name = ("reference",
                      "degraded",
                      "filter_ftype",
                      "gamma",
                      "size")
        param_types = (str, str, int, float, int)

        params = {
            p_name: p_type(input().strip())
            for p_name, p_type in zip(param_name, param_types)
        }

        filter_type = params.get("filter_type")

        if filter_type == 1:
            params["mode"] = input().strip()

        else:
            params["sigma"] = float(input().strip())

        return params

    model = ImageRestoration(**get_parameters())
    model.restore()
    rmse = model.compare()

    print("{:.3f}".format(rmse))
