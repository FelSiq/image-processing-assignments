import numpy as np

import image_base


class LogTransformation(image_base.ImageManipulatorBase):
    def __init__(self, img):
        super().__init__(img)

    def transform_img(self, c=None, epsilon=1.0e-8):
        """."""
        if self.img is None:
            raise TypeError("Please fit image first")

        img_max = self.img.max()

        if c is None:
            c = img_max / (np.log2(1 + img_max) + epsilon)

        self.img_mod = (c * np.log2(1 + np.abs(self.img))).astype(np.uint8)

        return self.img_mod


if __name__ == "__main__":
    import imageio
    import matplotlib.pyplot as plt

    import histogram_eq

    def fix_hist(hist, v_min, v_max):
        if v_min > 0 or v_max < 256:
            return np.concatenate((np.zeros(v_min),
                                  hist,
                                  np.zeros(255 - v_max)))
        return hist

    img1 = imageio.imread("./images/nap.jpg")
    img2 = imageio.imread("./images/scarlett.jpg")

    md1 = LogTransformation(img1)
    md2 = LogTransformation(img2)

    img1_t = md1.transform_img()
    img2_t = md2.transform_img()

    md1.plot()
    md2.plot()

    hist_img1 = histogram_eq.HistEqualiser._build_img_hist(img1)
    hist_img2 = histogram_eq.HistEqualiser._build_img_hist(img2)
    hist_img1_t = histogram_eq.HistEqualiser._build_img_hist(img1_t)
    hist_img2_t = histogram_eq.HistEqualiser._build_img_hist(img2_t)

    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.title("Image 1 Histogram")
    plt.ylabel("Frequency")
    plt.xlabel("Values")
    new_hist = fix_hist(hist_img1, img1.min(), img1.max())
    new_hist /= sum(new_hist)
    plt.gca().set_ylim((0, 0.05))
    plt.bar(range(256), new_hist)

    plt.subplot(222)
    plt.title("Modified Image 1 Histogram")
    plt.ylabel("Frequency")
    plt.xlabel("Values")
    new_hist = fix_hist(hist_img1_t, img1_t.min(), img1_t.max())
    new_hist /= sum(new_hist)
    plt.gca().set_ylim((0, 0.05))
    plt.bar(range(256), new_hist)

    plt.subplot(223)
    plt.title("Image 2 Histogram")
    plt.ylabel("Frequency")
    plt.xlabel("Values")
    new_hist = fix_hist(hist_img2, img2.min(), img2.max())
    new_hist /= sum(new_hist)
    plt.gca().set_ylim((0, 0.05))
    plt.bar(range(256), new_hist)

    plt.subplot(224)
    plt.title("Modified Image 2 Histogram")
    plt.ylabel("Frequency")
    plt.xlabel("Values")
    new_hist = fix_hist(hist_img2_t, img2_t.min(), img2_t.max())
    new_hist /= sum(new_hist)
    plt.gca().set_ylim((0, 0.05))
    plt.bar(range(256), new_hist)

    plt.show()
