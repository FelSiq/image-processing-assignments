import matplotlib.pyplot as plt
import numpy as np
import imageio


class HistEqualisation:
    def __init__(self, image):
        self.image = image
        self.image_eq = None
        self.img_min_val = None

    def hist_equalisation(self):
        """."""
        if self.image is None:
            raise TypeError("Image not fitted into model.")

        self.img_min_val = self.image.min()
        self.img_max_val = self.image.max()
        num_levels = self.img_max_val - self.img_min_val + 1

        hist = np.zeros(num_levels)

        for val, freq in zip(*np.unique(self.image, return_counts=True)):
            hist[val - self.img_min_val] = freq

        self.hist_eq = ((num_levels - 1.0) / self.image.size) * hist.cumsum()

        return self.hist_eq

    def transform_img(self):
        """."""
        if self.image is None:
            raise TypeError("Image not fitted into model.")

        self.image_eq = self.hist_eq[self.image - self.img_min_val]

        return self.image

    def plot_equalisation(self):
        if self.hist_eq is None:
            raise TypeError("Please call 'hist_equalisation' first.")

        plt.plot(range(self.img_min_val, self.img_max_val+1), self.hist_eq)
        plt.title("Histogram Equalisation Curve")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")
        plt.show()

    def plot(self, cmap="gray", size=(10, 10), num_bits=8):
        """."""
        if self.image is None:
            raise TypeError("Image not fitted into model.")

        plt.figure(figsize=size)

        if self.image_eq is not None:
            plt.subplot(121)

        plt.imshow(self.image, cmap=cmap, vmin=0, vmax=2**num_bits)
        plt.title("Original Image")

        if self.image_eq is not None:
            plt.subplot(122)
            plt.imshow(self.image_eq, cmap=cmap, vmin=0, vmax=2**num_bits)
            plt.title("Transformed Image")

        plt.show()


if __name__ == "__main__":
    md1 = HistEqualisation(imageio.imread("../images/nap.jpg"))
    md2 = HistEqualisation(imageio.imread("../images/scarlett.jpg"))

    md1.hist_equalisation()
    md2.hist_equalisation()

    md1.plot_equalisation()
    md2.plot_equalisation()

    md1.transform_img()
    md2.transform_img()

    md1.plot()
    md2.plot()
