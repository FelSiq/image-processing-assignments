"""Image Processing assignment from course SCC0251.

Assignment data:
    Title: "Assignment 2: Image Enhancement and Filtering"
    Year: 2019 (Semester 01 / Fall)

Student data:
    Name: Felipe Alves Siqueira
    NoUSP: 9847706
    Undergraduation Student
"""
import typing as t

import numpy as np
import imageio
import matplotlib.pyplot as plt


class ImageEnhancer:

    def __init__(self,
                 image: np.ndarray,
                 method: str,
                 threshold: t.Optional[float] = None,
                 filter_size: t.Optional[int] = None,
                 filter_weights: t.Optional[np.ndarray] = None):
        """Enhance an image using various techniques."""
        self.image = image
        self.transformed_image = None
        self.method = method
        self.threshold = threshold
        self.filter_weights = filter_weights
        self.filter_size = filter_size

    def limiarization(self) -> np.ndarray:
        """Find the optimal threshold to separate pixels in an image."""
        if not isinstance(self.threshold, (float, int)):
            raise TypeError("'threshold' must be integer or float, "
                            "received type {}".format(type(self.threshold)))

        prev_threshold = self.threshold + 0.5
        while abs(prev_threshold - self.threshold) >= 0.5:
            mean_1 = self.image[self.image > self.threshold].mean()
            mean_2 = self.image[self.image <= self.threshold].mean()

            prev_threshold = self.threshold
            self.threhsold = 0.5 * (mean_1 + mean_2)

        self.transformed_image = (self.image > self.threshold).astype(np.uint8)

        return self.transformed_image

    def filtering_1d(self) -> np.ndarray:
        """Apply a 1D filter in image."""
        if not isinstance(self.filter_weights, np.ndarray):
            raise TypeError("'filter weights' must be an np.ndarray, "
                            "got {}".format(type(self.filter_weights)))

        if not isinstance(self.filter_size, (float, int)):
            raise TypeError("'filter_size' must be integer or float, "
                            "received type {}".format(type(self.filter_size)))

        img_shape = self.image.shape

        flatten_copy = self.image.ravel()

        self.transformed_image = np.zeros(self.image.size).astype(float)

        half_filter_size = self.filter_size // 2

        for i in range(self.transformed_image.size):
            image_piece = np.take(
                flatten_copy,
                range(i - half_filter_size, i + half_filter_size + 1),
                mode="wrap")

            self.transformed_image[i] = np.dot(image_piece,
                                               self.filter_weights)

        self.transformed_image = self.transformed_image.reshape(img_shape)

        return self.transformed_image

    def median_filter(self):
        """Apply median filtering in the image."""
        if not isinstance(self.filter_size, (float, int)):
            raise TypeError("'filter_size' must be integer or float, "
                            "received type {}".format(type(self.filter_size)))

        hfs = self.filter_size // 2

        num_row, num_col = self.image.shape

        padded_img = np.pad(
            self.image,
            pad_width=hfs,
            mode="constant",
            constant_values=0,
        )

        self.transformed_image = np.array([
            [np.median(padded_img[(x - hfs):(x + hfs + 1),
                                  (y - hfs):(y + hfs + 1)])
             for y in range(hfs, num_col + hfs)]
            for x in range(hfs, num_col + hfs)
        ])

        return self.transformed_image

    def transform(self):
        if self.method == "limiarization":
            self.limiarization()

        elif self.method == "1D filtering":
            self.filtering_1d()

        elif self.method == "2D filtering":
            self.filtering_2d()

        elif self.method == "2D median":
            self.median_filter()

        else:
            raise ValueError("Unknown method.")

    def plot(self):
        plt.figure(figsize=(12, 12))
        plt.subplot(121)
        plt.title("Reference")
        plt.imshow(self.image, cmap="gray")

        plt.subplot(122)
        plt.title("Transformed")
        plt.imshow(self.transformed_image, cmap="gray")

        plt.show()

    def rescale(self,
                val_min: t.Union[int, float] = 0,
                val_max: t.Union[int, float] = 255,
                dtype: t.Type = np.uint8):
        if not isinstance(self.transformed_image, np.ndarray):
            raise TypeError("First 'transform' the fitted image "
                            "before rescaling.")

        img = self.transformed_image.astype(float)

        img_min = img.min()
        img_max = img.max()

        if img_min == img_max:
            return np.full(img.shape, val_min)

        img = (
            val_min +
            (val_max - val_min) * np.divide(img - img_min, img_max - img_min))

        self.transformed_image = img.astype(dtype)
        return self.transformed_image

    def compare(self) -> float:
        if not isinstance(self.image, np.ndarray):
            raise TypeError("'image' attribute type must be a np.ndarray, "
                            "got {}".format(type(self.image)))

        if self.transformed_image is None:
            raise TypeError("First call 'transform' before comparing the "
                            "transformed image.")

        num_row, num_col = self.image.shape

        flt_original_img = self.image.astype(float)
        flt_modified_img = self.transformed_image.astype(float)

        rmse = np.sqrt(
            np.power(flt_original_img - flt_modified_img, 2).sum()
            / (num_row * num_col))

        return rmse


if __name__ == "__main__":
    subpath = "./documents/Casos_Teste"

    def get_parameters(subpath: str = None):
        """Get test cases arguments."""
        def get_str(dtype: t.Type = str, split: bool = False) -> t.Any:
            if split:
                return np.array(input().strip().split()).astype(dtype)

            return dtype(input().strip())

        METHOD_NAME = {
            1: "limiarization",
            2: "1D filtering",
            3: "2D filtering",
            4: "2D median",
        }

        filepath = get_str()
        if subpath:
            filepath = "/".join((subpath, filepath))

        args = {
            "image": imageio.imread(filepath),
            "method": METHOD_NAME.get(get_str(dtype=int)),
        }

        method = args["method"]

        if method == "limiarization":
            args["threshold"] = get_str(dtype=float)

        elif method == "1D filtering":
            n = args["filter_size"] = get_str(dtype=int)
            args["filter_weights"] = get_str(dtype=float, split=True)

        elif method == "2D filtering":
            n = args["filter_size"] = get_str(dtype=int)

            args["filter_weights"] = np.array([
                get_str(dtype=float, split=True)
                for _ in range(n)
            ])

            args["threshold"] = get_str(dtype=float)

        elif method == "2D median":
            n = args["filter_size"] = get_str(dtype=int)

        else:
            raise ValueError("Unknown method.")

        return args

    model = ImageEnhancer(**get_parameters(subpath=subpath))
    model.transform()
    model.rescale(val_min=model.image.min(), val_max=model.image.max())
    rmse = model.compare()

    print("{:.4f}".format(rmse))

    model.plot()
