import numpy as np


class Fourier:
    """."""
    def __init__(self):
        """."""
        self.fourier_amplitudes = None

    def discrete_transform_1d(self,
                              signal: np.ndarray) -> np.ndarray:
        """."""
        N = signal.size

        self.fourier_amplitudes = np.zeros(N, dtype=np.complex)

        coef = -np.pi * 2.0j

        time_steps = np.arange(N)

        for freq in np.arange(N):
            self.fourier_amplitudes[freq] = np.sum(
                signal * np.exp(coef * freq * time_steps / N))

        self.fourier_amplitudes /= np.sqrt(N)

        return self.fourier_amplitudes

    def discrete_transform_2d(self,
                              image: np.ndarray) -> np.ndarray:
        """."""
        M, N = image.shape

        coef = -np.pi * 2.0j

        self.fourier_amplitudes = np.zeros((M, N), dtype=np.complex)

        for u in np.arange(M):
            for v in np.arange(N):
                exp_mat = np.fromfunction(
                    lambda x, y: np.exp(coef * (u*x/M + v*y/N)),
                    shape=image.shape)

                self.fourier_amplitudes[u, v] = np.multiply(
                        image, exp_mat).sum()

        self.fourier_amplitudes /= np.sqrt(M * N)

        return self.fourier_amplitudes


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    model = Fourier()
    time_steps = np.arange(0, 2, 0.002)

    signal = (1 * np.sin(4 * 2 * np.pi * time_steps)
              + 0.6 * np.cos(16 * 2 * np.pi * time_steps)
              + 0.2 * np.cos(7 * 2 * np.pi * time_steps))

    model.discrete_transform_1d(signal)
    plt.plot(time_steps, signal)
    plt.plot(time_steps, np.log(1.0 + np.abs(model.fourier_amplitudes)))
    plt.show()

    N = model.fourier_amplitudes.size
    print("Top 5 Fourier coefficients with greatest absolute value:")
    sorted_fourier_coefs = sorted(
        zip(model.fourier_amplitudes[:(N//2)], np.arange(N//2)),
        key=lambda item: abs(item[0]), reverse=True)

    total_energy = 0.0
    for coef, _ in sorted_fourier_coefs[:5]:
        total_energy += abs(coef)

    for pack in sorted_fourier_coefs[:5]:
        coef, position = pack
        print("{:<{fill}}: real (cossine): {:<{fill}.4f}  "
              "img (sine): {:.4f}  (partial energy: {:.4f}) "
              "".format(position,
                        coef.real,
                        coef.imag,
                        abs(coef) / total_energy,
                        fill=5))

    """
    import imageio

    def plot_graphs(img: np.ndarray, model: Fourier, img_id: int):
        plt.subplot(2, 3, 1 + 3 * img_id)
        plt.imshow(img[:size, :size], cmap="gray")
        plt.subplot(2, 3, 2 + 3 * img_id)
        plt.imshow(np.abs(model.fourier_amplitudes), cmap="gray")
        plt.subplot(2, 3, 3 + 3 * img_id)
        plt.imshow(np.log(1.0 + np.abs(model.fourier_amplitudes)), cmap="gray")

    model = Fourier()

    size = 32

    img1 = imageio.imread("images/gradient_noise.png")[:size, :size]
    img2 = imageio.imread("images/sin1.png")[:size, :size]

    model.discrete_transform_2d(img1)
    plot_graphs(img1, model, 0)
    model.discrete_transform_2d(img2)
    plot_graphs(img2, model, 1)

    plt.show()
    """
