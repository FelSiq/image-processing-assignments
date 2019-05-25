"""Image Processing assignment from course SCC0251.

Assignment data:
    Title: "Assignment 3: image restoration"
    Year: 2019 (Semester 01 / Fall)

Student data:
    Name: Felipe Alves Siqueira
    NoUSP: 9847706
    Undergraduation Student
"""
import typing as t

import numpy as np
import imageio
import scipy.fftpack


class ImageRestoration:
    def __init__(self,
                 img_reference: np.ndarray,
                 img_degraded: np.ndarray,
                 filter_type: int,
                 gamma: float,
                 size: int,
                 mode: t.Optional[str] = None,
                 sigma: t.Optional[float] = None):
        """Implements two technique of image reconstruction.

        Args
        ----
            img_reference : :obj:`np.ndarray`
                Image to be used as reference, used to calculate the
                RMSE (Root Mean Squared Error).

            img_degraded : :obj:`np.ndarray`
                Degraded image to be reconstructed.

            filter_type : :obj:`int`
                Type of technique used to reconstruct the image. Must
                be 1 or 2.
                    * 1: Adaptive Denoising.
                    * 2: Constrained Least Squares.

            gamma : :obj:`float`
                A real number in 0 and 1 (both inclusive) interval.
                Used as adjust factor in both image reconstruction
                methods.

            size : :obj:`int`
                Filter size used to reconstruct the image.

            mode : :obj:`str`, optional
                If ``filter_type`` is 1 (Adaptive Denoising), this
                option is used to select the strategy used to estimate
                values necessary to reconstruct the image. This
                parameter must assume one value between ``robust``
                and ``average``.

            sigma : :obj:`float`, optional
                If ``filter_type`` is 2 (Constrained Least Squares), this
                parameter is used to construct the Gaussian Kernel. Must
                assume positive values.
        """
        if filter_type not in (1, 2):
            raise ValueError("Unknown filter type '{}'. "
                             "Expecting 1 or 2.".format(filter_type))

        if mode is not None and mode not in ("average", "robust"):
            raise ValueError("Unknown mode parameter '{}'. Expecting "
                             "'average' or 'robust'".format(mode))

        if sigma is not None and sigma <= 0:
            raise ValueError("sigma must be a positive number. Got "
                             "{}".format(sigma))

        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1] interval. Got "
                             "{}".format(gamma))

        if not ((mode is None) ^ (sigma is None)):
            raise TypeError("'mode' and 'sigma' can't be both None or "
                            "both given. Choose precisely one.")

        self.img_ref = self._open_img(img_reference)
        self.img_deg = self._open_img(img_degraded)

        self.img_restored = None

        self.gamma = gamma
        self.size = size
        self.filter_type = filter_type
        self.sigma = sigma
        self.mode = mode

        self._laplacian = np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0],
        ])

    def _open_img(self, filepath: str) -> np.ndarray:
        """Load the image from the given filepath."""
        return imageio.imread(filepath).astype(float)

    def normalize_img(self,
                      v_min: t.Union[int, float] = 0,
                      v_max: t.Union[int, float] = 255) -> np.ndarray:
        """Normalize the image values to [v_min, v_max] interval."""
        if self.img_restored is None:
            raise TypeError("Please restore an image before normalizing it.")

        img_min = self.img_restored.min()
        img_max = self.img_restored.max()

        if img_min == img_max:
            self.img_restored = np.full(self.img_restored.shape, v_min)
            return self.img_restored

        self.img_restored = (v_min + (v_max - v_min)
                             * (self.img_restored - img_min)
                             / (img_max - img_min))

        return self.img_restored

    def adaptive_denoising(self,
                           window_prop: float = 1.0/6.0,
                           epsilon: float = 1.0e-8,
                           normalize: bool = True) -> np.ndarray:
        """Apply Adaptive Denoising in the fitted image.

        Args
        ----
            window_prop : :obj:`float`
                Proportion of the image, for each dimension, to be used
                as the sample window to calculate the global dispersion
                of the image.

            epsilon : :obj:`float`
                A tiny value. Every number with absolute value equal or
                less than it will be considered zero.

            normalize : :obj:`float`
                If True, then the outputed image will be normalized
                using the minimum and maximum value of the degenerated
                image.

        Returns
        -------
        :obj:`np.ndarray`
            Reconstructed image using Adaptive Denoising technique.
        """
        if not 0 < window_prop <= 1:
            raise ValueError("window_prop must be in (0, 1] interval.")

        def calculate_disp_n(
                window_prop: float,
                epsilon: float = 1.0e-8,
                ) -> float:
            """Calculate sample dispersion ``disp_n``."""
            sample_rows, sample_cols = np.floor(np.array(self.img_deg.shape)
                                                * window_prop).astype(int)

            image_sample = self.img_deg[:sample_rows, :sample_cols]

            if self.mode == "average":
                disp_n = np.std(image_sample)

            else:
                disp_n = np.subtract(*np.percentile(image_sample, (75, 25)))

            return disp_n if abs(disp_n) >= epsilon else 1.0

        def apply_filter(disp_n: float,
                         epsilon: float = 1.0e-8,
                         ) -> t.Tuple[np.ndarray, np.ndarray]:
            """Get centrality and dispersion measure for the image."""
            centr_l = np.copy(self.img_deg)
            disp_l = np.full(self.img_deg.shape, fill_value=disp_n)

            num_row, num_col = self.img_deg.shape
            half_size = self.size // 2

            for i in np.arange(half_size, num_row-half_size):
                for j in np.arange(half_size, num_col-half_size):
                    img_sample = self.img_deg[(i-half_size):(i+half_size+1),
                                              (j-half_size):(j+half_size+1)]

                    if self.mode == "average":
                        centr_l[i, j] = np.mean(img_sample)
                        cur_disp = np.std(img_sample)

                    else:
                        q25, median, q50 = np.percentile(img_sample, (25, 50, 75))
                        centr_l[i, j] = median
                        cur_disp = q50 - q25

                    if abs(cur_disp) >= epsilon:
                        disp_l[i, j] = cur_disp

            return disp_l, centr_l

        disp_n = calculate_disp_n(
            window_prop=window_prop,
            epsilon=epsilon)

        disp_l, centr_l = apply_filter(
            disp_n=disp_n,
            epsilon=epsilon)

        self.img_restored = (self.img_deg - self.gamma
                             * (disp_n / disp_l)
                             * np.round(self.img_deg - centr_l))

        if normalize:
            self.normalize_img(
                v_min=self.img_deg.min(),
                v_max=self.img_deg.max())

        return self.img_restored

    def _gaussian_filter(self, k: int = 3, sigma: float = 1.0):
        """Generate a Gaussian Filter with dimension k."""
        arx = np.arange((-k // 2) + 1.0, (k // 2) + 1.0)
        x, y = np.meshgrid(arx, arx)
        filt = np.exp(-0.5 * (np.square(x) + np.square(y)) / np.square(sigma))
        return filt / np.sum(filt)

    def _pad(self,
             img: np.ndarray,
             output_shape: t.Tuple[int, int],
             constant: int = 0) -> np.ndarray:
        """Pad (constant) the given image to the desired ``output_shape``."""
        pad_row, pad_col = ((output_shape - np.array(img.shape)) // 2)
        out_img = np.full(output_shape, constant, dtype=np.complex)

        num_row, num_col = np.array(img.shape)

        out_img[pad_row:(pad_row + num_row),
                pad_col:(pad_col + num_col)] = img

        return out_img

    def power_spectrum(self, img: np.ndarray) -> np.ndarray:
        """Calculate the power spectrum of the given image."""
        return np.power(np.abs(img), 2)

    def constrained_lsf(self, normalize: bool = False) -> np.ndarray:
        """Apply Constrained Least Squared in the fitted image."""
        def laplacian_power_spec(
                img_deg: np.ndarray,
                gamma: float) -> np.ndarray:
            """Calculate the weighted Laplacian Power Spectrum."""
            lap_power_spec = self._pad(
                img=self._laplacian,
                output_shape=img_deg.shape)

            lap_power_spec = gamma * self.power_spectrum(
                scipy.fftpack.fft2(lap_power_spec))

            return lap_power_spec

        def degradation_function(
                img_deg: np.ndarray,
                filter_size: int,
                sigma: float) -> t.Tuple[np.ndarray, np.ndarray]:
            """Calculate the degradation function Fourier Transform."""
            deg_func = self._gaussian_filter(
                k=filter_size,
                sigma=sigma)

            deg_func_ft = self._pad(
                img=deg_func,
                output_shape=img_deg.shape)

            deg_func_ft = scipy.fftpack.fft2(deg_func_ft)

            deg_func_power_spec = self.power_spectrum(deg_func_ft)

            return deg_func_ft, deg_func_power_spec

        laplacian_coeff = laplacian_power_spec(
            img_deg=self.img_deg,
            gamma=self.gamma)

        deg_func_ft, deg_func_power_spec = degradation_function(
            img_deg=self.img_deg,
            filter_size=self.size,
            sigma=self.sigma)

        img_deg_ft = scipy.fftpack.fft2(self.img_deg)

        self.img_restored = scipy.fftpack.ifft2(np.multiply(
            deg_func_ft.conj() / (deg_func_power_spec + laplacian_coeff),
            img_deg_ft)).real

        if normalize:
            self.normalize_img(
                v_min=self.img_deg.min(),
                v_max=self.img_deg.max())

        self.img_restored = np.roll(
            a=self.img_restored,
            shift=np.array(self.img_restored.shape) // 2,
            axis=(0, 1))

        return self.img_restored

    def restore(self) -> np.ndarray:
        """Restore the fitted image using the chosen ``filter_type``."""
        if self.img_deg is None:
            raise TypeError("Please load a degenerated image "
                            "before restorating.")
        """Apply the selected image restoration technique."""
        if self.filter_type == 1:
            # Denoising
            return self.adaptive_denoising()

        elif self.filter_type == 2:
            # Deblurring
            return self.constrained_lsf()

        else:
            raise ValueError("Invalid filter type. Got '{}', "
                             "expected 1 or 2.".format(self.filter_type))

    def compare(self) -> float:
        """Return RMSE between the restored and reference images."""
        if self.img_ref is None or self.img_restored is None:
            raise TypeError("Please load a reference image and call "
                            "'restore' method before comparing.")

        img_a = self.img_restored.astype(float)
        img_b = self.img_ref.astype(float)
        rmse = np.sqrt(np.power(img_a - img_b, 2.0).sum()
                       / np.prod(self.img_restored.shape))
        return rmse


if __name__ == "__main__":
    subpath = None
    # subpath = "./tests/"

    def get_parameters(subpath: str = None) -> t.Dict[str, t.Any]:
        """Get test case parameters."""
        param_name = ("img_reference",
                      "img_degraded",
                      "filter_type",
                      "gamma",
                      "size")
        param_types = (str, str, int, float, int)

        params = {
            p_name: p_type(input().strip())
            for p_name, p_type in zip(param_name, param_types)
        }

        filter_type = params.get("filter_type")

        if filter_type == 1:
            params["mode"] = input().strip()

        else:
            params["sigma"] = float(input().strip())

        if subpath:
            params["img_reference"] = subpath + params["img_reference"]
            params["img_degraded"] = subpath + params["img_degraded"]

        return params

    model = ImageRestoration(**get_parameters(subpath=subpath))
    model.restore()
    rmse = model.compare()

    print("{:.3f}".format(rmse))
