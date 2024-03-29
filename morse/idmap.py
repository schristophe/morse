#
#
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy

from .auxil import *
from .echellediagram import *
from .eigenvalue import *
from .pattern import *
from .spectrum import *


class IDMap(object):
    """Class to represent a DFT map.

    Attributes:
        spectrum (Spectrum): Frequency data.
        m (int): Azimuthal order.
        k (int): Ordering index (Lee & Saio 97).
        nurot_vect (np.array): Rotation frequencies tested (in µHz).
        f0_vect (np.array): Frequencies at which the DFT is evaluated (in µHz).
        folded (bool):
            If True, it is assumed that the spectrum is folded in the
            inertial frame.
        eigenvalue (Eigenvalue):
            Eigenvalue of Laplace's tidal equation corresponding to (m,k).
        resolution (np.array):
        dft_map (np.array):
        psd_max (float): Maximum of Power Spectral Density (PSD) in dft_map.
        nurot (float):
            Rotation frequency corresponding to the maximum of PSD in dft_map
            (in µHz).
        buoy_r (float):
            Buoyancy radius corresponding to the maximum of PSD in dft_map
            (in s).
        psd_threshold (float):
            Detection threshold.
        flag_detection (bool):
            True if a period spacing pattern has been detected (psd_max >
            psd_threshold).
        echelle_diagram (EchelleDiagram):
            Echelle diagram made using the parameters (nurot, buoy_r)
            that maximise the PSD.
        tol (float):
            Maximum relative period difference tolerated to associate an
            observed mode to one of the TAR model.
        offset (float): Offset.
        err_periods_mc (float):
            Mean period difference between a TAR model and observed
            modes.
        results_mc (np.array):
            Results of the MC simulation. A line contains nurot,
            buoy_r, psd_max and flag_detection for a draw of the
            perturbed spectrum.
        err_nurot (float):
            Standard deviation of the perturbed rotation frequencies (in µHz).
        err_buoy_r (float):
            Standard deviation of the perturbed buoyancy radii (in s).
    """

    def __init__(self, spectrum, m, k, nurot_vect, f0_vect, folded=False):
        """Initialises an instance of IDMap.

        Args:
            spectrum (Spectrum object): Frequency data.
            m (int): Azimuthal order.
            k (int): Ordering index (Lee & Saio 97).
            nurot_vect (np.array): Rotation frequencies tested (in µHz)
            f0_vect (np.array):
                Frequencies at which the DFT is evaluated (in µHz).
            folded (bool):
                If True, it is assumed that the spectrum is folded in the
                inertial frame.

        """
        self.spectrum = spectrum
        self.m = m
        self.k = k
        self.nurot_vect = nurot_vect
        self.f0_vect = f0_vect
        self.folded = folded
        self.eigenvalue = Eigenvalue(m, k)
        # Initialising the ID map and the array that will store the resolution
        # of each DFT spectrum
        resolution = np.nan * np.ones(len(nurot_vect))
        dft_map = np.nan * np.ones((len(nurot_vect), len(f0_vect)))
        # For each value of nurot, stretches the oscillation spectrum and
        # computes its DFT
        nurot_vect = nurot_vect / FACTOR_ROT  # converting to c/d for computations
        for i in np.arange(len(nurot_vect)):
            progress_bar(i + 1, len(nurot_vect), "Computing the ID map...")
            nurot = nurot_vect[i]
            periods_co = in2co(spectrum.periods, m, k, nurot, folded)
            if len(periods_co) > 3:
                stretched_periods = stretch(m, k, periods_co, self.eigenvalue, nurot)
                resolution[i] = 1.0 / (
                    np.max(stretched_periods) - np.min(stretched_periods)
                )
                # Compute the DFT of the stretched spectrum
                dft_map[i, :] = _dft(
                    stretched_periods, f0_vect / FACTOR_ROT
                )  # f0_vect has to be in c/d
        self.resolution = resolution * FACTOR_ROT
        self.dft_map = dft_map
        # Looking for which parameters (nurot,f0) maximise the PSD in dft_map
        self.psd_max = np.nanmax(self.dft_map)
        imax_nurot, imax_f0 = np.argwhere(self.dft_map == self.psd_max)[0]
        self.nurot = self.nurot_vect[imax_nurot]
        self.buoy_r = (
            1e6 / f0_vect[imax_f0]
        )  # we want the buoyancy radius (dim of a time)
        # Computing the detection threshold
        # (above which we consider we have detected a period spacing pattern)
        nb_bins = np.ceil(
            (self.f0_vect.max() - self.f0_vect.min()) / self.resolution[imax_nurot]
        )
        self.psd_threshold = -np.log(1 - 0.99 ** (1 / nb_bins))
        if self.psd_max > self.psd_threshold:
            self.flag_detection = True
        else:
            self.flag_detection = False

    def get_echelle_diagram(self):
        """Computes the echelle diagram using the parameters that maximise the
        PSD in the DFT map.

        """
        self.echelle_diagram = EchelleDiagram(
            self.spectrum,
            self.m,
            self.k,
            self.nurot,
            self.buoy_r,
            folded=self.folded,
        )

    def get_pattern(self, tol=1e-3):
        """If `flag_detection` is True, extracts the detected period spacing pattern.

        Args:
            tol (float):
                Maximum relative difference tolerated between modelled and observed
                periods of the pattern modes.
        Returns:
            Pattern: Extracted period spacing pattern.

        """
        if self.flag_detection:
            return Pattern(
                self.spectrum,
                self.m,
                self.k,
                self.nurot,
                self.buoy_r,
                self.folded,
                tol,
            )

    def calc_uncertainties(self, ndraws=500, propagate=False, tol=5e-3):
        """Estimates the uncertainty on the rotation frequency and buoyancy
        radius using a Monte-Carlo simulation.

        Args:
            ndraws (int): Number of draws.
            propagate (bool):
                If True, propagates uncertainties on mode periods. Otherwise,
                the mean period difference between a TAR model and observed
                modes is used as a "fictive" uncertainty.
            tol (float):
                Only used if propagate is False. Maximum relative period
                difference tolerated to associate an observed mode to one of the
                TAR model.
        """
        if propagate == True and hasattr(
            self.spectrum, "errs"
        ):  # propagate errors on mode periods
            period_errs = self.spectrum.errs / self.spectrum.errs ** 2
        else:  # or the mean difference between TAR model and observed modes
            self.tol = tol
            # Get the mean offset
            nurot = self.nurot / FACTOR_ROT  # µHz -> c/d
            buoy_r = self.buoy_r / 86400.0  # s -> d
            self.offset = get_offset(
                self.spectrum,
                self.m,
                self.k,
                nurot,
                buoy_r,
                folded=self.folded,
            )
            # Generate a synthetic spectrum within the asymptotic TAR
            synth_spectrum = Spectrum()
            synth_spectrum.generate(
                self.m, self.k, self.nurot, self.buoy_r, offset=self.offset
            )
            # Filter synthetic modes that are not in spectrum
            filt_synth_spectrum = synth_spectrum.filter(
                periodmin=(1 - 5 * tol) * min(self.spectrum.periods),
                periodmax=(1 + 5 * tol) * max(self.spectrum.periods),
            )
            # For each observed mode, associate the nearest synthetic mode if the period difference is inferior to tol
            synth_periods = np.array([])
            matched_obs_periods = np.array([])
            filter_obs_spectrum = np.array([])
            for i in np.arange(len(self.spectrum.periods)):
                i_closest = np.argmin(
                    abs(filt_synth_spectrum.periods - self.spectrum.periods[i])
                )
                if (
                    abs(
                        filt_synth_spectrum.periods[i_closest]
                        - self.spectrum.periods[i]
                    )
                    / self.spectrum.periods[i]
                    <= tol
                ):
                    synth_periods = np.append(
                        synth_periods, filt_synth_spectrum.periods[i_closest]
                    )
                    filter_obs_spectrum = np.append(filter_obs_spectrum, True)
                else:
                    filter_obs_spectrum = np.append(filter_obs_spectrum, False)
                    print(
                        "No synthetic mode corresponds to observed period = "
                        + str("%.3f" % self.spectrum.periods[i])
                        + " d"
                    )
            matched_obs_spectrum = self.spectrum.filter(boolean=filter_obs_spectrum)
            self.err_periods_mc = np.std(matched_obs_spectrum.periods - synth_periods)
        # Adjust the resolution and bounds of the DFT map so the computing time stays reasonable
        i_bounds = np.argwhere(self.dft_map >= 0.40 * self.psd_max)
        nurot_vect_dg = np.linspace(
            self.nurot_vect[i_bounds[:, 0]].min(),
            self.nurot_vect[i_bounds[:, 0]].max(),
            100,
        )
        f0_vect_dg = np.linspace(
            self.f0_vect[i_bounds[:, 1]].min(), self.f0_vect[i_bounds[:, 1]].max(), 100
        )
        results = np.zeros((ndraws, 4))
        for i in np.arange(ndraws):
            progress_bar(
                i + 1,
                ndraws,
                status="Evaluating uncertainty on estimated parameters...",
            )
            # Draw a spectrum by perturbing mode periods/frequencies
            perturbed_spectrum = deepcopy(matched_obs_spectrum)
            perturbed_spectrum.periods = np.random.normal(
                loc=perturbed_spectrum.periods, scale=self.err_periods_mc
            )
            # Compute the DFT map for the perturbed spectrum
            perturbed_idmap = IDMap(
                perturbed_spectrum,
                self.m,
                self.k,
                nurot_vect_dg,
                f0_vect_dg,
                self.folded,
            )
            # Store results
            results[i, :] = np.array(
                [
                    perturbed_idmap.nurot,
                    perturbed_idmap.buoy_r,
                    perturbed_idmap.psd_max,
                    perturbed_idmap.flag_detection,
                ]
            )
        self.results_mc = results
        self.err_nurot = np.std(results[:, 0])
        self.err_buoy_r = np.std(results[:, 1])

    def plot(self, cmap="cividis", save=False):
        """Plots the computed DFT map.

        Args:
            cmap (str): Colour map to use for the plot.
            save (bool): If True, saves the figure to a png file.

        """
        plt.figure()
        plt.tick_params(
            axis="both", direction="inout", which="major", top=False, right=False
        )
        plt.tick_params(
            axis="both", direction="inout", which="minor", top=False, right=False
        )
        plt.pcolormesh(
            self.f0_vect,
            self.nurot_vect,
            self.dft_map,
            cmap=cmap,
            shading="nearest",
            rasterized=True,
        )
        plt.xlabel(r"$1/P_0$ ${\rm (\mu Hz)}$")
        plt.ylabel(r"$\nu_{\rm rot}$ ${\rm (\mu Hz)}$")
        plt.xlim([self.f0_vect.min(), self.f0_vect.max()])
        plt.ylim([self.nurot_vect.min(), self.nurot_vect.max()])
        cb = plt.colorbar(fraction=0.045, pad=0.04)
        cb.set_label(r"$|DFT(\sqrt{\lambda_{m,k}}P_{\rm co})|^2$")
        # Plot contours
        niveau = np.nanmax(self.dft_map) * np.array([0.50, 0.95])
        contours = [(0, (10, 5)), "solid"]
        plt.contour(
            self.f0_vect,
            self.nurot_vect,
            self.dft_map,
            levels=niveau,
            linestyles=contours,
            colors="k",
        )
        if save == True:
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
                + "IDMAP_m"
                + str("%d" % self.m)
                + "_k"
                + str("%d" % self.k)
                + ".png"
            )
            plt.savefig(filename)
        plt.show()

    def save(self):
        """Saves results, log and plots."""
        if hasattr(self.spectrum, "path"):
            filename = self.spectrum.path.split("/")[-1] + "_"
        else:
            filename = ""
        if not os.path.isdir(os.getcwd() + "/results/"):
            os.mkdir(os.getcwd() + "/results/")
        # Save results
        fname = (
            os.getcwd()
            + "/results/"
            + filename
            + "IDMAP-RESULTS_m"
            + str("%d" % self.m)
            + "_k"
            + str("%d" % self.k)
            + ".txt"
        )
        with open(fname, "w") as fres:
            if hasattr(self, "err_nurot"):
                fres.write(
                    f"{'detection':10s}\t{self.flag_detection}\n"
                    + f"{'m':10s}\t{self.m:10d}\n"
                    + f"{'k':10s}\t{self.k:10d}\n"
                    + f"{'nurot':10s}\t{self.nurot:10.4f}\n".format("nurot", self.nurot)
                    + f"{'err_nurot':10s}\t{self.err_nurot:10.4f}\n"
                    + f"{'P0':10s}\t{self.buoy_r:10.4f}\n"
                    + f"{'err_P0':10s}\t{self.err_buoy_r:10.4f}\n"
                    + f"{'max_PSD':10s}\t{self.psd_max:10.2f}\n"
                )
            else:
                fres.write(
                    f"{'detection':10s}\t{self.flag_detection}\n"
                    + f"{'m':10s}\t{self.m:10d}\n"
                    + f"{'k':10s}\t{self.k:10d}\n"
                    + f"{'nurot':10s}\t{self.nurot:10.4f}\n"
                    + f"{'P0':10s}\t{self.bouyancy_radius:10.4f}\n"
                    + f"{'max_PSD':10s}\t{self.psd_max:10.2f}\n"
                )
        # Save results of the MC simulation (if applicable)
        if hasattr(self, "err_nurot"):
            fname = (
                os.getcwd()
                + "/results/"
                + filename
                + "IDMAP-RESULTS-MC_m"
                + str("%d" % self.m)
                + "_k"
                + str("%d" % self.k)
                + ".txt"
            )
            hdr = ("{:10s}\t" * 4).format("nurot", "P0", "max_PSD", "detection")
            format = "%10.4f " * 3 + "%10d"
            np.savetxt(fname, self.results_mc, header=hdr, fmt=format)
        # Save log (everything to reproduce the results)
        fname = (
            os.getcwd()
            + "/results/"
            + filename
            + "IDMAP-LOG_m"
            + str("%d" % self.m)
            + "_k"
            + str("%d" % self.k)
            + ".txt"
        )
        with open(fname, "w") as flog:
            flog.write("# SPECTRUM\n")
            if hasattr(self.spectrum, "amps"):
                flog.write(
                    f"ampmin = {min(self.spectrum.amps):8g}\n"
                    + f"ampmax = {max(self.spectrum.amps):8g}\n"
                    + f"freqmin = {min(self.spectrum.freqs):8g}\n"
                    + f"freqmax = {max(self.spectrum.freqs):8g}\n"
                )
            flog.write(
                "\n# IDMAP\n"
                + f"m = {self.m:d}\n"
                + f"k = {self.k:d}\n"
                + f"scan_nurot = ({min(self.nurot_vect):g},{max(self.nurot_vect):g},{len(self.nurot_vect):d})\n"
                + f"scan_f0 = ({min(self.f0_vect):g},{max(self.f0_vect):g},{len(self.f0_vect):d})\n"
                + f"folded = {self.folded}\n"
            )
            if hasattr(self, "err_nurot") and hasattr(self, "offset"):
                flog.write(
                    "\n# IDMAP MC SIMULATION\n"
                    + "propagate = model_err\n"
                    + f"ndraws = {len(self.results_mc):d}\n"
                    + f"tol = {self.tol:8g}\n"
                    + f"offset = {self.offset:8g}\n"
                )
            elif hasattr(self, "err_nurot"):
                flog.write(
                    "\n# IDMAP MC SIMULATION\n"
                    + "propagate = obs_err\n"
                    + f"ndraws = {len(self.results_mc):d}\n"
                )
        # Save plots
        self.plot(save=True)
        if not hasattr(self, "echelle_diagram"):
            self.get_echelle_diagram()
        self.echelle_diagram.plot(save=True)
        plt.close("all")


# Auxiliary functions used by IDmap methods


def _dft(P, f0_vect):
    """Computes the Discrete Fourier Transform (DFT) of the spectrum which
    modes have for periods P and for amplitudes one.

    Args:
        P (np.array): Mode periods (in d).
        f0_vect (np.array): Frequencies at which the DFT is evaluated (in c/d).

    Returns:
        np.array: The Power Spectral Density (PSD).

    """
    N = len(P)
    M = f0_vect.reshape((len(f0_vect), 1)) @ P.reshape((1, N))
    dft = (1 / N) * (
        np.sum(np.cos(2 * np.pi * M), axis=1) ** 2
        + np.sum(np.sin(2 * np.pi * M), axis=1) ** 2
    )
    return dft
