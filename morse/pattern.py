#
#
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from .auxil import *
from .eigenvalue import *
from .spectrum import *


class Pattern(object):
    """Class to represent a period spacing pattern.

    Attributes:
        spectrum (Spectrum):
            The period spacing is extracted from the frequency data in spectrum.
        m (int): Azimuthal order.
        k (int): Ordering index (Lee & Saio 97).
        nurot (float): Rotation frequency (in µHz).

            Determined by fitting a TAR model to the observed period spacing pattern.

        err_nurot (float): Error on fitted nurot (µHz).
        buoy_r (float): Buoyancy radius (in s).

            Determined by fitting a TAR model to the observed period spacing pattern.

        err_buoy_r (float): Error on fitted buoyancy radius (s).
        offset (float): Offset.

            It is assumed that the fitted nurot and buoy_r are exact. Offset
            varies widely with slighly different parameters therefore you should not
            attach any physical meaning to it.

        folded (bool):
            If True, it is assumed that the spectrum is folded in the inertial frame.
        tol (float):
            Maximum relative difference tolerated between modelled and observed mode
            periods.
        model (Spectrum):
            Best-fit TAR model saved as a Spectrum object containing the pulsation modes
            from n = 1 to 150.
        freqs_obs (np.array): Observed mode frequencies (in µHz).
        periods_obs (np.array): Observed mode periods (in d).
        spacings_obs (np.array): Observed period spacings (in d).
        eta_obs (np.array): Observed spin parameters.
        freqs_mod (np.array): Modelled mode frequencies (in µHz).
        periods_mod (np.array): Modelled mode periods (in d).
        spacings_mod (np.array): Modelled period spacings (in d).
        eta_mod (np.array): Modelled spin parameters.
        err_freqs (np.array): Uncertainties on mode frequencies (in µHz).
        err_periods (np.array): Uncertainties on mode periods (in d).
        err_spacings (np.array): Uncertainties on period spacings (in d).
    """

    def __init__(
        self,
        spectrum,
        m,
        k,
        nurot,
        buoy_r,
        folded=False,
        tol=1e-3,
        max_miss=5,
    ):
        """Initialises an instance of Pattern.

        A period spacing pattern of (m,k) modes is extracted from the observed
        `spectrum`, assuming it follows the asymptotic TAR.

        Args:
            spectrum (Spectrum):
                Spectrum from which the period spacing will be extracted.
            m (int): Azimuthal order.
            k (int): Ordering index (Lee & Saio 97).
            nurot (float): Initial guess on the rotation frequency (in µHz).
            buoy_r (float): Initial guess on the buoyancy radius (in s).
            folded (bool):
                If True, it is assumed that the spectrum is folded in the
                inertial frame.
            tol (float):
                Maximum relative difference tolerated between modelled and observed
                mode periods.
            max_miss (int): Maximum number of consecutive missing modes.
        """
        self.spectrum = spectrum
        self.m = m
        self.k = k
        self.nurot = nurot
        self.buoy_r = buoy_r
        self.folded = folded
        self.tol = tol

        # Get the mean offset
        self.offset = get_offset(
            spectrum, m, k, nurot / FACTOR_ROT, buoy_r / 86400, folded
        )
        self.fit()

    def fit(self, cov=False):
        """Fits the period spacing pattern with the asymptotic TAR.

        Args:
            cov (bool): If True, uses the full covariance matrix (not implemented yet).
        """
        # Create
        model = Spectrum()
        model.generate(self.m, self.k, self.nurot, self.buoy_r, offset=self.offset)
        matched = self.spectrum.match(model, self.tol)
        matched.sort()

        # Only select consecutive orders for the fit
        i_consecutive = np.argwhere(abs(np.diff(matched.n2)) == 1)[:, 0]
        periods_to_fit = matched.periods[i_consecutive]
        period_spacings_to_fit = np.diff(matched.periods)[i_consecutive]

        # Compute covariance matrix if needed (TODO: Regularisation of the cov. matrix)
        # if cov and hasattr(self.spectrum, "errs"):
        #     # Propagation of frequency errors on periods
        #     period_errs = self.spectrum.periods ** 2 * (self.spectrum.errs /
        #            FACTOR_ROT)
        #     # cov(DeltaP(n),DeltaP(j+1)) = -var(P(j+1))
        #     supdiag = -period_errs[1:-1] ** 2
        #     # var(DeltaP(n)) = var(P(j)) + var(P(j+1))
        #     diag = period_errs[1:] ** 2 + period_errs[:-1] ** 2
        #     covmat = np.diag(supdiag, k=-1) + np.diag(diag) + np.diag(supdiag, k=1)
        #     covmat = covmat[i_consecutive, :][:, i_consecutive]
        #     self.covmat = covmat
        # else:
        #     # equivalent to identity matrix
        #     covmat = None
        covmat = None

        # Construct the parametric model to feed into curve_fit
        tar_model = _func_tar_model(
            self.nurot,
            self.buoy_r,
            self.offset,
            self.m,
            self.k,
        )

        popt, pcov = curve_fit(
            tar_model,
            periods_to_fit,
            period_spacings_to_fit,
            p0=[self.nurot, self.buoy_r],
            bounds=([0, 50], [50, 20000]),
            sigma=covmat,
        )
        # Update parameters
        self.nurot = popt[0]
        self.buoy_r = popt[1]
        self.offset = get_offset(
            self.spectrum,
            self.m,
            self.k,
            self.nurot / FACTOR_ROT,
            self.buoy_r / 86400,
            self.folded,
        )

        # Store parameter uncertainties
        pstd = np.sqrt(np.diag(pcov))
        self.err_nurot = pstd[0]
        self.err_buoy_r = pstd[1]

        # Store best TAR model
        self.model = Spectrum()
        self.model.generate(self.m, self.k, self.nurot, self.buoy_r, offset=self.offset)
        self.model.sort()

        # Store period spacing pattern data
        pattern = self.spectrum.match(self.model, self.tol)
        pattern.sort()
        self.n = pattern.n2
        nb_modes = len(self.n)
        # Observations
        self.freqs_obs = pattern.freqs
        self.periods_obs = pattern.periods
        consecutive = abs(np.diff(pattern.n2)) == 1
        self.spacings_obs = np.full(nb_modes, np.nan)
        self.spacings_obs[:-1] = np.where(
            consecutive, np.diff(pattern.periods) * 86400, np.nan
        )
        self.eta_obs = (
            2
            * (self.nurot / FACTOR_ROT)
            * in2co(self.periods_obs, self.m, self.k, self.nurot / FACTOR_ROT)
        )
        if hasattr(pattern, "errs"):
            self.err_freqs = pattern.errs
            self.err_periods = pattern.periods ** 2 * (pattern.errs / FACTOR_ROT)
            self.err_spacings = np.full(nb_modes, np.nan)
            self.err_spacings[:-1] = np.where(
                consecutive,
                np.sqrt(np.cumsum(self.err_periods ** 2)[1:]) * 86400,
                np.nan,
            )
        # TAR model
        self.freqs_mod = pattern.freqs2
        self.periods_mod = pattern.periods2
        self.spacings_mod = np.full(nb_modes, np.nan)
        self.spacings_mod[:-1] = np.where(
            consecutive, np.diff(pattern.periods2) * 86400, np.nan
        )
        self.eta_mod = (
            2
            * (self.nurot / FACTOR_ROT)
            * in2co(self.periods_mod, self.m, self.k, self.nurot / FACTOR_ROT)
        )

    def plot(self, save=False):
        """Plots the period spacing pattern.

        Args:
            save (bool): If True, saves the figure to a png file.
        """
        plt.figure()

        # Plot the best TAR model
        tar_model = _func_tar_model(
            self.nurot, self.buoy_r, self.offset, self.m, self.k
        )
        plt.plot(
            self.periods_obs,
            tar_model(
                self.periods_obs,
                self.nurot,
                self.buoy_r,
            )
            * 86400,
        )
        # Represent the uncertainties in fitted parameters
        # Note: nurot and buoy_r are highly correlated, this is probably not
        # the proper way to do it.
        if self.k == -2 and self.m == 1:
            plt.fill_between(
                self.periods_obs,
                tar_model(
                    self.periods_obs,
                    self.nurot - self.err_nurot,
                    self.buoy_r - self.err_buoy_r,
                )
                * 86400,
                tar_model(
                    self.periods_obs,
                    self.nurot + self.err_nurot,
                    self.buoy_r + self.err_buoy_r,
                )
                * 86400,
                alpha=0.3,
            )
        if self.k >= 0 and self.m <= 0:
            plt.fill_between(
                self.periods_obs,
                tar_model(
                    self.periods_obs,
                    self.nurot + self.err_nurot,
                    self.buoy_r - self.err_buoy_r,
                )
                * 86400,
                tar_model(
                    self.periods_obs,
                    self.nurot - self.err_nurot,
                    self.buoy_r + self.err_buoy_r,
                )
                * 86400,
                alpha=0.3,
            )

        # Plot observed spacings
        plt.errorbar(
            self.periods_obs,
            self.spacings_obs,
            yerr=self.err_spacings,
            fmt="o",
            color="black",
        )

        plt.xlabel("Period (d)")
        plt.ylabel("Period spacing (s)")

        if save:
            if hasattr(self.spectrum, "path"):
                filename = self.spectrum.path.split("/")[-1] + "_"
            else:
                filename = ""
            if not os.path.isdir(os.getcwd() + "/results/"):
                os.mkdir(os.getcwd() + "/results/")
            filename = (
                os.getcwd()
                + "/results/"
                + filename
                + "PATTERN_m"
                + str("%d" % self.m)
                + "_k"
                + str("%d" % self.k)
                + ".png"
            )
            plt.savefig(filename)
        plt.show()

    def save(self):
        """Saves information about the pattern (mode ID, rotation frequency,
        buoyancy radius, modes) and saves a plot of the pattern together with the
        best-fit model.
        """
        if hasattr(self.spectrum, "path"):
            filename = self.spectrum.path.split("/")[-1] + "_"
        else:
            filename = ""
        if not os.path.isdir(os.getcwd() + "/results/"):
            os.mkdir(os.getcwd() + "/results/")
        # Save individual modes constituting the pattern
        fname = (
            os.getcwd()
            + "/results/"
            + filename
            + "PATTERN-DATA_m"
            + str("%d" % self.m)
            + "_k"
            + str("%d" % self.k)
            + ".txt"
        )
        if hasattr(self, "err_freqs"):
            hdr = ("{:4s} " + "{:10s} " * 11).format(
                "n",
                "freqs_obs",
                "err_freqs",
                "P_obs",
                "err_P",
                "DeltaP_obs",
                "err_DeltaP",
                "eta_obs",
                "freqs_mod",
                "P_mod",
                "DeltaP_mod",
                "eta_mod",
            )
            format = "%4d " + "%10.6f " * 11
            data = np.column_stack(
                (
                    self.n,
                    self.freqs_obs,
                    self.err_freqs,
                    self.periods_obs,
                    self.err_periods,
                    self.spacings_obs,
                    self.err_spacings,
                    self.eta_obs,
                    self.freqs_mod,
                    self.periods_mod,
                    self.spacings_mod,
                    self.eta_mod,
                )
            )
        else:
            hdr = ("{:4s} " + "{:10s} " * 8).format(
                "n",
                "freqs_obs",
                "P_obs",
                "DeltaP_obs",
                "eta_obs",
                "freqs_mod",
                "P_mod",
                "DeltaP_mod",
                "eta_mod",
            )
            format = "%4d " + "%10.6f " * 8
            data = np.column_stack(
                (
                    self.n,
                    self.freqs_obs,
                    self.periods_obs,
                    self.spacings_obs,
                    self.eta_obs,
                    self.freqs_mod,
                    self.periods_mod,
                    self.spacings_mod,
                    self.eta_mod,
                )
            )
        np.savetxt(fname, data, header=hdr, fmt=format)
        # Save global data about the pattern
        fname = (
            os.getcwd()
            + "/results/"
            + filename
            + "PATTERN-GLOBAL_m"
            + str("%d" % self.m)
            + "_k"
            + str("%d" % self.k)
            + ".txt"
        )
        with open(fname, "w") as fglob:
            fglob.write(
                f"{'m':10s}\t{self.m:10d}\n"
                + f"{'k':10s}\t{self.k:10d}\n"
                + f"{'nurot':10s}\t{self.nurot:10.4f}\n"
                + f"{'err_nurot':10s}\t{self.err_nurot:10.4f}\n"
                + f"{'P0':10s}\t{self.buoy_r:10.4f}\n"
                + f"{'err_P0':10s}\t{self.err_buoy_r:10.4f}\n"
                + f"{'offset':10s}\t{self.offset:10.4f}\n"
                + f"{'nmodes':10s}\t{len(self.n):10d}\n"
                + f"{'nmin':10s}\t{self.n.min():10d}\n"
                + f"{'nmax':10s}\t{self.n.max():10d}"
            )
        self.plot(save=True)
        plt.close("all")


def _func_tar_model(nurot, buoy_r, offset, m, k):
    """Builds and returns a "continuous" function representing the period
    spacings as a function of periods by interpolating a discrete TAR model.

    Args:
        nurot (float): Rotation frequency (in µHz).
        buoy_r (float): Buoyancy radius (in s).
        offset (float): Offset.
        m (int): Azimuthal order.
        k (int): Ordering index (Lee & Saio 97).
    """

    def func_temp(periods, nurot, buoy_r):
        model = Spectrum()
        model.generate(m, k, nurot, buoy_r, nmax=150)
        model.sort()
        period_spacings = interp1d(
            model.periods[:-1],
            np.diff(model.periods),
            kind="quadratic",
            fill_value="extrapolate",
        )
        return period_spacings(periods)

    return func_temp
