import sys
sys.path.insert(0, "../")

import image_base  # noqa: E402


class ImageInverter(image_base.ImageManipulatorBase):
    def __init__(self, img):
        """Invert image pixel values using a simple formulation.

        pixel'[i, j] = (img_max + img_min) - pixel[i, j]
        """
        super().__init__(img)

    def transform_img(self):
        """."""
        if self.img is None:
            raise TypeError("Please fit an image in model first.")

        img_range = self.img.max() + self.img.min()

        self.img_mod = img_range - self.img

        return self.img_mod


if __name__ == "__main__":
    import imageio
    md1 = ImageInverter(imageio.imread("../images/nap.jpg"))
    md2 = ImageInverter(imageio.imread("../images/scarlett.jpg"))

    md1.transform_img()
    md2.transform_img()

    md1.plot()
    md2.plot()
