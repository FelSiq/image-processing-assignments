import matplotlib.pyplot as plt
import numpy as np

import image_base


class HistEqualiser(image_base.ImageManipulatorBase):
    def __init__(self, img):
        """Calculates a equalisation function from a given image.

        The equalisation function tries to distribute the levels of the
        image more uniformly.
        """
        super().__init__(img)

        self.img_min_val = None
        self.hist = None
        self.hist_eq = None

    @classmethod
    def _build_img_hist(cls, img):
        img_min_val = img.min()
        img_max_val = img.max()
        num_levels = img_max_val - img_min_val + 1

        hist = np.zeros(num_levels).astype(int)

        for val, freq in zip(*np.unique(img, return_counts=True)):
            hist[val - img_min_val] = freq

        return hist

    def hist_equalisation(self):
        """Calculate the equalisation function of an image with a histogram."""
        if self.img is None:
            raise TypeError("Image not fitted into model.")

        self.img_min_val = self.img.min()
        self.img_max_val = self.img.max()
        num_levels = self.img_max_val - self.img_min_val + 1

        self.hist = HistEqualiser._build_img_hist(self.img)

        self.hist_eq = (((num_levels - 1.0) / float(self.img.size))
                        * self.hist.cumsum()).astype(int)

        return self.hist_eq

    def transform_img(self):
        """Transform the image using the calculated equalisation function."""
        if self.img is None:
            raise TypeError("Image not fitted into model.")

        self.img_mod = self.hist_eq[self.img - self.img_min_val]

        return self.img

    def plot_equalisation(self):
        """Plot the calculated equalisation function."""
        if self.hist_eq is None:
            raise TypeError("Please call 'hist_equalisation' first.")

        val_range = range(self.img_min_val, self.img_max_val+1)

        plt.subplot(131)
        plt.title("Histogram Equalisation Curve")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")
        plt.plot(val_range, self.hist_eq)

        plt.subplot(132)
        plt.title("Original Image Histogram")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")
        plt.bar(val_range, self.hist)

        if self.img_mod is not None:
            plt.subplot(133)
            plt.title("Equalised Image Histogram")
            plt.xlabel("Intensity")
            plt.ylabel("Frequency")
            plt.bar(val_range,
                    HistEqualiser._build_img_hist(self.img_mod))

        plt.show()


if __name__ == "__main__":
    import imageio
    md1 = HistEqualiser(imageio.imread("./images/nap.jpg"))
    md2 = HistEqualiser(imageio.imread("./images/scarlett.jpg"))

    md1.hist_equalisation()
    md2.hist_equalisation()

    md1.transform_img()
    md2.transform_img()

    md1.plot_equalisation()
    md2.plot_equalisation()

    md1.plot()
    md2.plot()
