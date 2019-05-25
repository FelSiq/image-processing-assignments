"""Image Processing assignment from course SCC0251.

Assignment data:
    Title: "Assignment 4: Colour image processing and segmentation"
    Year: 2019 (Semester 01 / Fall)

Student data:
    Name: Felipe Alves Siqueira
    NoUSP: 9847706
    Undergraduation Student
"""
import typing as t

import numpy as np
import random
import imageio


class KMeans:
    """Naive implementation of K-Means clustering algorithm."""
    def __init__(self,
                 data: np.ndarray,
                 cluster_num: int = 3,
                 it_max: int = 1.0e+5,
                 random_seed: int = None):
        """."""
        self.X = data.copy()
        self.cluster_num = cluster_num
        self.it_max = it_max
        self.random_seed = random_seed

        self.cluster_ids = None
        self.centroids = None

    def _init_centroids(self):
        """Init model centroids."""
        num_inst, _ = self.X.shape

        random.seed(self.random_seed)
        centroid_ids = np.sort(
            random.sample(range(num_inst), self.cluster_num))

        self.centroids = self.X[centroid_ids, :]

    def fit(self, dist_ord: int = 2):
        """."""
        self._init_centroids()

        num_inst, _ = self.X.shape
        self.cluster_ids = np.zeros(num_inst)

        for _ in np.arange(self.it_max):
            # Update instance clusters once.
            for inst_id, inst_coord in enumerate(self.X):
                self.cluster_ids[inst_id] = np.argmin([
                    np.linalg.norm(centroid_coord - inst_coord, ord=2)
                    for centroid_coord in self.centroids
                ])

            # Update centroid coordinates once.
            self.centroids = np.array([
                self.X[self.cluster_ids == cl_id, :].mean(axis=0)
                for cl_id in np.arange(self.cluster_num)
            ])

        return self.cluster_ids


class ImageTransformer:
    """Various methods for image transformation."""
    def __init__(self, option: int):
        """."""
        if option not in np.arange(1, 5):
            raise ValueError("'option' must be an integer between 1 "
                             "and 4 (both inclusive).")

        self.option = option

        self._TRANSFORMATIONS = (
            self.transformation_rgb,
            self.transformation_rgbxy,
            self.transformation_luminance,
            self.transformation_luminancexy,
        )
        """Available transformations of this class."""

        self._LUMINANCE_MAGIC_NUMBERS = np.array([.299, .587, .114])
        """Magic numbers given by the assignment specification."""

    def _concat_coords(self,
                       img: np.ndarray,
                       num_row: int,
                       num_col: int) -> np.ndarray:
        """Concatenate (x, y) coordinates as two new attributes in data."""
        vals_x = np.array([[
            j for j in np.arange(num_col)]
            for i in np.arange(num_row)
        ]).reshape(-1, 1)

        vals_y = np.array([
            [i] * num_col
            for i in np.arange(num_row)
        ]).reshape(-1, 1)

        return np.hstack((img, vals_x, vals_y))

    def transformation_rgb(self, img: np.ndarray) -> np.ndarray:
        """Transform the given MxN image into a RGB (M*N)x3 dataset."""
        num_row, num_col, channels = img.shape

        if channels != 3:
            raise ValueError("Image must contain 3 color channels "
                             "to perform RGB transformation.")

        img = img.reshape(num_row * num_col, channels)

        return img

    def transformation_luminance(self, img: np.ndarray) -> np.ndarray:
        """Transform the given MxN image into a Luminance (M*N)x1 dataset."""
        img = np.apply_along_axis(
            arr=img,
            func1d=lambda pixel: np.dot(self._LUMINANCE_MAGIC_NUMBERS, pixel),
            axis=2).reshape(-1, 1)

        return img

    def transformation_rgbxy(self, img: np.ndarray) -> np.ndarray:
        """Transform the given MxN image into a RGBxy (M*N)x5 dataset."""
        num_row, num_col, _ = img.shape

        img = self._concat_coords(
            self.transformation_rgb(img),
            num_row=num_row,
            num_col=num_col)

        return img

    def transformation_luminancexy(self, img: np.ndarray) -> np.ndarray:
        """Transform the given MxN image into a Luminancexy (M*N)x3 dataset."""
        num_row, num_col, _ = img.shape

        img = self._concat_coords(
            self.transformation_luminance(img),
            num_row=num_row,
            num_col=num_col)

        return img

    def transform(self, img: np.ndarray) -> np.ndarray:
        """Transform the given image with given ``option``."""
        return self._TRANSFORMATIONS[self.option - 1](img)

    def normalize_img(self,
                      v_min: t.Union[int, float] = 0,
                      v_max: t.Union[int, float] = 255) -> np.ndarray:
        """Normalize the image values to [v_min, v_max] interval."""
        if self.img_seg is None:
            raise TypeError("Please restore an image before normalizing it.")

        img_min = self.img_seg.min()
        img_max = self.img_seg.max()

        if img_min == img_max:
            self.img_seg = np.full(self.img_seg.shape, v_min)
            return self.img_seg

        self.img_seg = (v_min + (v_max - v_min)
                        * (self.img_seg - img_min) / (img_max - img_min))

        return self.img_seg


class ImageColSeg(ImageTransformer):
    def __init__(self,
                 img_inp: str,
                 img_ref: str,
                 option: int,
                 cluster_num: int = 3,
                 it_max: int = 1.0e+5,
                 random_seed: t.Optional[int] = None):
        """

        Args
        ----
        """
        super().__init__(option=option)

        if not isinstance(it_max, int):
            raise TypeError("'it_max' must be an integer.")

        if not isinstance(cluster_num, int):
            raise TypeError("'cluster_num' must be an integer.")

        if cluster_num <= 0:
            raise ValueError("'cluster_num' must be a positive integer.")

        if it_max <= 0:
            raise ValueError("'it_max' must be a positive integer.")

        self.img_ref = self._open_img(img_ref, npy=True)
        self.img_inp = self.transform(self._open_img(img_inp))

        self.option = option

        self._kmeans_model = KMeans(
            data=self.img_inp,
            cluster_num=cluster_num,
            it_max=it_max,
            random_seed=random_seed)
        """Instantiate the K-Means model for coloring/segmentation."""

        self.img_seg = None

    def _open_img(self, filepath: str, npy: bool = False) -> np.ndarray:
        """Load the image from the given filepath."""
        if npy:
            return np.load(filepath).astype(float)

        return imageio.imread(filepath).astype(float)

    def compare(self) -> float:
        """Return RMSE between the restored and reference images."""
        if self.img_seg is None or self.img_ref is None:
            raise TypeError("Please load a reference image and call "
                            "'restore' method before comparing.")

        img_a = self.img_seg.astype(float)
        img_b = self.img_ref.astype(float)
        rmse = np.sqrt(np.power(img_a - img_b, 2.0).sum()
                       / np.prod(self.img_seg.shape))
        return rmse

    def colseg(self, normalize: bool = True) -> np.ndarray:
        """Colorize and segment the image using K-Means."""
        self.img_seg = self._kmeans_model.fit()

        if normalize:
            self.normalize_img()

        self.img_seg = self.img_seg.reshape(self.img_ref.shape[:2])

        return self.img_seg


if __name__ == "__main__":
    subpath = None
    # subpath = "./tests/"

    def get_parameters(subpath: str = None) -> t.Dict[str, t.Any]:
        """Get test case parameters."""
        param_name = ("img_inp",
                      "img_ref",
                      "option",
                      "cluster_num",
                      "it_max",
                      "random_seed")
        param_types = (str, str, int, int, int, int)

        params = {
            p_name: p_type(input().strip())
            for p_name, p_type in zip(param_name, param_types)
        }

        return params

    model = ImageColSeg(**get_parameters(subpath=subpath))
    model.colseg()
    rmse = model.compare()

    print("{:.4f}".format(rmse))
