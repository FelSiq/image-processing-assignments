import numpy as np

import image_base


class ImgThresholder(image_base.ImageManipulatorBase):
    def __init__(self, img):
        """Set image values to 1 if higher than a threshold, 0 otherwise."""
        super().__init__(img)

    def transform_img(self, threshold: int = 127) -> np.ndarray:
        """Set image values to 1 if higher than a threshold, 0 otherwise."""
        if self.img is None:
            raise TypeError("Please fit an image in the model first.")

        self.img_mod = np.zeros(self.img.shape).astype(np.uint8)

        self.img_mod[self.img > threshold] = 1

        return self.img_mod


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import imageio

    md = ImgThresholder(imageio.imread("./images/scarlett.jpg"))

    plt.figure(figsize=(12, 12))

    for plt_index, threshold in enumerate(range(0, 256, 16)):
        md.transform_img(threshold=threshold)
        plt.subplot(4, 4, 1+plt_index)
        plt.imshow(md.img_mod, vmin=0, vmax=1, cmap="gray")

    plt.show()
