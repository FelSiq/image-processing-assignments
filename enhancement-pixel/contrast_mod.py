import sys
sys.path.insert(0, "../")

import image_base  # noqa: E402


class ContrastModulation(image_base.ImageManipulatorBase):
    def __init__(self, img):
        """Change value range of given image to an new interval."""
        super().__init__(img)

    def transform_img(self, interval=(0, 1)):
        """Change value range of given image to an new interval."""
        if self.img is None:
            raise TypeError("Please fit an image in the model first.")

        img_min = self.img.min()
        img_max = self.img.max()

        int_min, int_max = interval

        self.img_mod = (int_min + (self.img - img_min)
                        * ((int_max - int_min) / (img_max - img_min)))

        return self.img_mod


if __name__ == "__main__":
    import imageio
    md1 = ContrastModulation(imageio.imread("../images/nap.jpg"))
    md2 = ContrastModulation(imageio.imread("../images/scarlett.jpg"))

    img1 = md1.transform_img(interval=(50, 200))
    img2 = md2.transform_img(interval=(51, 211))

    print(img1.min(), img1.max())
    print(img2.min(), img2.max())

    md1.plot()
    md2.plot()
