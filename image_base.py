import matplotlib.pyplot as plt


class ImageManipulatorBase:
    def __init__(self, img):
        self.img = img
        self.img_mod = None

    def plot(self, cmap="gray", size=(10, 10), num_bits=8):
        """."""
        if self.img is None:
            raise TypeError("Image not fitted into model.")

        plt.figure(figsize=size)

        if self.img_mod is not None:
            plt.subplot(121)

        plt.imshow(self.img, cmap=cmap, vmin=0, vmax=2**num_bits)
        plt.title("Original Image")

        if self.img_mod is not None:
            plt.subplot(122)
            plt.imshow(self.img_mod, cmap=cmap, vmin=0, vmax=2**num_bits)
            plt.title("Transformed Image")

        plt.show()
