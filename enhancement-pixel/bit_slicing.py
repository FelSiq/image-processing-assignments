import sys

import numpy as np

sys.path.insert(0, "../")

import image_base  # noqa: E402


class ImgThresholder(image_base.ImageManipulatorBase):
    def __init__(self, img):
        """Get a single specific bit from eahc image pixel."""
        super().__init__(img)

    def transform_img(self, bit: int = 1) -> np.ndarray:
        """Get a single specific bit from eahc image pixel."""
        if self.img is None:
            raise TypeError("Please fit an image in the model first.")

        if not 0 < bit <= 8:
            raise ValueError("'bit' argument must be an integer between 1 "
                             "and 8 (inclusive)")

        self.img_mod = self.img & 1 << (bit - 1)

        return self.img_mod


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import imageio
    img = imageio.imread("../images/scarlett.jpg")

    md = ImgThresholder(img)

    plt.figure(figsize=(12, 12))

    plt.subplot(3, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    for bit in range(8, 0, -1):
        md.transform_img(bit=bit)
        plt.subplot(3, 3, 10 - bit)
        plt.imshow(md.img_mod, cmap="gray")
        plt.title("Bit: {}".format(bit))
        plt.axis("off")

    plt.show()
