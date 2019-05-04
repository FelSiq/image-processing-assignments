import typing as t

import numpy as np


class Fourier:
    """Toy implementation of Fourier transformation and related methods."""

    def __init__(self):
        """."""
        self.fourier_amplitudes = None
        self.total_energy = 0.0
        self.top_fourier_coeffs = None

    def discrete_transform_1d(self, signal: np.ndarray) -> np.ndarray:
        """1-D Slow (complexity O(N**2)) Fourier transform of given signal."""
        N = signal.size

        self.fourier_amplitudes = np.zeros(N, dtype=np.complex)

        coef = -np.pi * 2.0j

        time_steps = np.arange(N)

        for freq in np.arange(N):
            self.fourier_amplitudes[freq] = np.sum(
                signal * np.exp(coef * freq * time_steps / N))

        self.fourier_amplitudes /= np.sqrt(N)

        return self.fourier_amplitudes

    def inverse_transform_1d(self) -> np.ndarray:
        """Reverse the Fourier transformation of fitted 1-D coefficients."""
        N = self.fourier_amplitudes.size

        signal = np.zeros(N, dtype=np.complex)

        coef = np.pi * 2.0j

        time_steps = np.arange(N)

        for freq in np.arange(N):
            signal[freq] = np.sum(
                 self.fourier_amplitudes
                 * np.exp(coef * freq * time_steps / N))

        signal /= np.sqrt(N)

        return signal.real

    def _fft_1d(self, signal: np.ndarray) -> np.ndarray:
        """."""
        if signal.size <= 1:
            return signal

        sig_even = self._fft_1d(signal[::2])
        sig_odd = self._fft_1d(signal[1::2])

        const_coef = -np.pi * 2.0j / signal.size
        fourier_coeffs = np.zeros(signal.size, dtype=np.complex)

        for freq in np.arange(signal.size // 2):
            coef = np.exp(const_coef * freq)
            fourier_coeffs[freq] = sig_even[freq] + coef * sig_odd[freq]
            fourier_coeffs[freq + signal.size // 2] = (sig_even[freq]
                                                       - coef * sig_odd[freq])

        return fourier_coeffs

    def fft_1d(self, signal: np.ndarray) -> np.ndarray:
        """1-D Fast (time comp. O(NlogN)) Fourier transform of given signal."""
        if np.log2(signal.size) % 1 > 0.0:
            raise ValueError("This implementation only supports signals with"
                             "size of powers of 2.")

        self.fourier_amplitudes = self._fft_1d(signal) / np.sqrt(signal.size)

        return self.fourier_amplitudes

    def discrete_transform_2d(self, image: np.ndarray) -> np.ndarray:
        """2-D Slow (complexity O(N**2)) Fourier transform of given image."""
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

    def inverse_transform_2d(self) -> np.ndarray:
        """Reverse the Fourier transformation of fitted 2-D coefficients."""
        M, N = self.fourier_amplitudes.shape

        coef = np.pi * 2.0j

        image = np.zeros((M, N), dtype=np.complex)

        for u in np.arange(M):
            for v in np.arange(N):
                exp_mat = np.fromfunction(
                    lambda x, y: np.exp(coef * (u*x/M + v*y/N)),
                    shape=self.fourier_amplitudes.shape)

                image[u, v] = np.multiply(
                        self.fourier_amplitudes, exp_mat).sum()

        image /= np.sqrt(M * N)

        return image.real

    def top_k_coeffs(self, k: int = 5) -> t.Tuple[np.ndarray, np.ndarray]:
        """Return the k Fourier coefficients with greatest absolute value."""
        N = self.fourier_amplitudes.size

        self.top_fourier_coeffs = sorted(
            zip(self.fourier_amplitudes[:(N//2)], np.arange(N//2)),
            key=lambda item: abs(item[0]), reverse=True)[:k]

        self.total_energy = 0.0
        for coef, _ in self.top_fourier_coeffs:
            self.total_energy += abs(coef)

        return self.top_fourier_coeffs

    def phase_angle(self, epsilon: float = 1.0e-10):
        """Return the Phase Angle of the Fourier Coefficients.

        The Phase Angle Phi is defined as

            Phi(x) = arctan(Real(F(x)) / Imag(F(x)))

        Where F is the Fourier Coefficients, Real(.) is the Real
        part of a complex number and Imag(.) is the imaginary
        part of a complex number.
        """
        return np.arctan(self.fourier_amplitudes.imag
                         / (epsilon + self.fourier_amplitudes.real))

    def power_spectrum(self):
        """Return the Power Spectrum of the Fourier Coefficients.

        The Power Spectrum P is defined as P(x) = abs(F(x))**2.0,
        where F is the Fourier Coefficients.
        """

        return np.power(np.abs(self.fourier_amplitudes), 2)

    def shift(self):
        """Shift in-place the Fourier coefficients to the center of array."""
        self.fourier_amplitudes = np.roll(
            a=self.fourier_amplitudes,
            shift=np.array(self.fourier_amplitudes.shape) // 2,
            axis=np.arange(len(self.fourier_amplitudes.shape)))

        return self.fourier_amplitudes

    def __repr__(self):
        my_str = []
        for coeff_item in self.top_fourier_coeffs:
            coef, position = coeff_item
            my_str.append(
                "{:<{fill}}: real (cossine): {:<{fill}.4f}  "
                "img (sine): {:.4f}  (partial energy: {:.4f}) "
                "".format(position,
                          coef.real,
                          coef.imag,
                          abs(coef) / self.total_energy,
                          fill=5))

        return "\n".join(my_str)


if __name__ == "__main__":
    """
    # 1-D signal experiment
    import matplotlib.pyplot as plt

    model = Fourier()
    time_steps = np.arange(0, 2.048, 0.002)

    signal = (1 * np.sin(4 * 2 * np.pi * time_steps)
              + 0.6 * np.cos(16 * 2 * np.pi * time_steps)
              + 0.2 * np.cos(7 * 2 * np.pi * time_steps))

    amp_slow = model.discrete_transform_1d(signal)
    amp_fast = model.fft_1d(signal)

    plt.plot(time_steps, signal)
    plt.plot(time_steps, np.log(1.0 + np.abs(amp_slow)))
    plt.plot(time_steps, np.log(1.0 + np.abs(amp_fast)))
    plt.show()

    print("Top 5 Fourier coefficients with greatest absolute value:")
    model.top_k_coeffs()
    print(model)

    rec_signal = model.inverse_transform_1d()

    plt.plot(time_steps, signal)
    plt.plot(time_steps, rec_signal)
    plt.show()

    plt.plot(time_steps, model.phase_angle())
    plt.title("Phase Angle")
    plt.show()

    plt.plot(time_steps, model.power_spectrum())
    plt.title("Power Spectrum")
    plt.show()
    """

    """
    # 2-D signal (image) experiment
    import matplotlib.pyplot as plt
    import imageio

    def plot_graphs(img: np.ndarray,
                    rec_img: np.ndarray,
                    model: Fourier,
                    img_id: int):
        plt.subplot(2, 4, 1 + 4 * img_id)
        plt.title("Original img")
        plt.imshow(img[:size, :size], cmap="gray")

        plt.subplot(2, 4, 2 + 4 * img_id)
        plt.title("Reconstructed img")
        plt.imshow(rec_img, cmap="gray")

        plt.subplot(2, 4, 3 + 4 * img_id)
        plt.title("Fourier Coeffs")
        plt.imshow(np.abs(model.fourier_amplitudes), cmap="gray")

        plt.subplot(2, 4, 4 + 4 * img_id)
        plt.title("Log Fourier Coeffs")
        plt.imshow(np.log(1.0 + np.abs(model.fourier_amplitudes)), cmap="gray")

    model = Fourier()

    size = 8

    img1 = imageio.imread("images/gradient_noise.png")[:size, :size]
    img2 = imageio.imread("images/sin1.png")[:size, :size]

    img1_ampl = model.discrete_transform_2d(img1).copy()
    rec_img1 = model.inverse_transform_2d()
    model.shift()
    plot_graphs(img1, rec_img1, model, 0)

    img2_ampl = model.discrete_transform_2d(img2).copy()
    rec_img2 = model.inverse_transform_2d()
    model.shift()
    plot_graphs(img2, rec_img2, model, 1)

    plt.show()
    """
