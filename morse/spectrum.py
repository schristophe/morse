#
#
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from scipy.interpolate import interp1d
from sklearn.cluster import DBSCAN

from .auxil import *
from .eigenvalue import *


class Spectrum(object):
    """Class to represent frequency data either extracted from the
    oscillation spectrum of a star or generated using the equation of the
    traditional approximation of rotation in its asymptotic formulation.

    Attributes:
        path (str): Path to the file containing the frequency data.
        freqs (np.array): Frequencies (in µHz).
        periods (np.array): Periods (in d).
        errs (np.array): Errors on frequencies (in µHz).
        amps (np.array): Amplitudes.
        m (int): Azimutal order.
        k (int): Ordering index (Lee & Saio 97).
        nurot (float): Rotation frequency (in µHz).
        buoyancy_radius (float): Buoyancy radius (in s).
        offset (float): Offset.
        n (np.array): Radial orders.
        periods_co (np.array): Periods in the co-rotating frame (in d).

    """

    def __init__(self):
        """Initialises an instance of Spectrum"""

    def load(self, path, colfreqs=0, colerrs=-1, colamps=-1):
        """Loads frequency data from a file.

        The file must contain the mode frequencies in µHz at the very least.
        Peak amplitudes and errors on frequencies may also be loaded. By
        default, only frequencies are loaded.

        Args:
            path (str): Path to the file containing the frequency data.
            colfreqs (int): Column number of frequencies.
            colerrs (int): Column number of frequency errors.
            colamps (int): Column number of mode amplitudes.

        """
        self.path = path
        data = np.genfromtxt(self.path)
        self.freqs = data[:, colfreqs]
        self.periods = FACTOR_ROT / self.freqs
        if colerrs != -1:
            self.errs = data[:, colerrs]
        if colamps != -1:
            self.amps = data[:, colamps]

    def generate(self, m, k, nurot, buoyancy_radius, offset=0.0, nmin=1, nmax=90):
        """Generates synthetic frequency data using the asymptotic formulation
        of the traditional approximation of rotation (TAR).

        Args:
            m (int): Azimuthal order.
            k (int): Ordering index (Lee & Saio 97).
            nurot (float): Rotation frequency (in µHz).
            buoyancy_radius (float): Buoyancy radius (in s).
            offset (float): Offset.
            nmin (int): Minimum radial order.
            nmax (int): Maximum radial order.

        """
        self.m = m
        self.k = k
        self.nurot = nurot
        self.buoyancy_radius = buoyancy_radius
        self.offset = offset
        if k + abs(m) > 0:
            n = np.arange(nmin, nmax + 1, 1)
        elif k + abs(m) <= 0:
            n = np.arange(nmin, nmax + 1, 1)
        self.n = n
        nurot = nurot / FACTOR_ROT  # nurot has to be in c/d
        buoyancy_radius = buoyancy_radius / 86400.0  # buoyancy_radius in d
        if nurot > 0:
            eigenvalue = Eigenvalue(m, k)
            P = eigenvalue.eta / (2 * nurot)
            f_P_k = buoyancy_radius / np.sqrt(eigenvalue.lamb)
            periods_co = np.array([])
            for i in n:
                f_P = f_P_k * (i + offset)
                periods_co = np.append(periods_co, find_zeros(P, f_P - P, 9))
        else:
            if k + abs(m) > 0:
                l = k + abs(m)
                periods_co = buoyancy_radius * (n + offset) / np.sqrt(l * (l + 1))
            elif k + abs(m) <= 0 and m != 0:
                sys.exit(
                    "(k = "
                    + str(k)
                    + ", m = "
                    + str(m)
                    + ") modes does not exist without rotation."
                )
            else:
                sys.exit("Radial modes are not handled in this version of the code.")
        self.periods_co = periods_co
        self.periods = co2in(periods_co, m, nurot)
        self.freqs = FACTOR_ROT / self.periods  # in µHz

    def plot(self):
        """Plots the oscillation spectrum."""
        plt.figure()
        if hasattr(self, "amps"):
            plt.vlines(self.freqs, 0, self.amps, lw=1)
        else:
            plt.vlines(self.freqs, 0, 1, lw=1)
        plt.xlabel("Frequency (µHz)")
        plt.ylabel("Amplitude")
        plt.show()

    def search_combinations(self, order=2):
        """ """

    def filter(
        self,
        ampmin=0,
        ampmax=np.inf,
        freqmin=0,
        freqmax=np.inf,
        periodmin=0,
        periodmax=np.inf,
        nmin=0,
        nmax=np.inf,
        boolean=None,
    ):
        """Filters the spectrum according to the period/frequency/amplitude of
        the modes or in an arbitrary way.

        Only modes that satify all conditions are kept (logical and).

        Args:
            ampmin (float): Minimum amplitude.
            ampmax (float): Maximum amplitude.
            freqmin (float): Minimum mode frequency.
            freqmax (float): Maximum mode frequency.
            periodmin (float): Minimum mode period.
            periodmax (float): Maximum mode period.
            nmin (int): Minimum radial order.
            nmax (int): Maximum radial order.
            boolean (np.array): Custom mask.

        Returns:
            Spectrum: The filtered spectrum.

        """
        # Make the filter
        if boolean is not None:
            conditions = boolean
        else:
            conditions = np.ones(len(self.periods))
        if hasattr(self, "amps"):
            conditions = np.vstack(
                (conditions, np.array([self.amps >= ampmin, self.amps <= ampmax]))
            )
        if hasattr(self, "n"):
            conditions = np.vstack(
                (conditions, np.array([self.n >= nmin, self.n <= nmax]))
            )
        conditions = np.vstack(
            (
                conditions,
                np.array(
                    [
                        self.freqs >= freqmin,
                        self.freqs <= freqmax,
                        self.periods >= periodmin,
                        self.periods <= periodmax,
                    ]
                ),
            )
        )
        filter = np.logical_and.reduce(conditions)
        # Creating a new Spectrum object with modes that pass the filter
        filtered_spectrum = deepcopy(self)
        filtered_spectrum.periods = filtered_spectrum.periods[filter]
        filtered_spectrum.freqs = filtered_spectrum.freqs[filter]
        if hasattr(filtered_spectrum, "amps"):
            filtered_spectrum.amps = filtered_spectrum.amps[filter]
        if hasattr(filtered_spectrum, "n"):
            filtered_spectrum.n = filtered_spectrum.n[filter]
        if hasattr(filtered_spectrum, "errs"):
            filtered_spectrum.errs = filtered_spectrum.errs[filter]
        if hasattr(filtered_spectrum, "periods_co"):
            filtered_spectrum.periods_co = filtered_spectrum.periods_co[filter]
        return filtered_spectrum

    def clustering(self, eps, min_samples):
        """Automatically detects frequency groups using the scikit-learn
        DBSCAN clustering algorithm.

        Args:
            eps (float):
                Maximum distance between two peak frequencies for one to be
                considered in the neighbourhood of the other.
            min_samples (int):
                Minimum number of peak frequencies in a neighbourhood for a
                given peak frequency to be considered as a core point (including
                the frequency peak considered).

        Returns:
            frequency_groups (np.array):
                Array of Spectrum objects representing the frequency groups
                detected by the clustering algorithm. They are ordered by
                increasing frequency.
        """
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(self.freqs.reshape(-1, 1))
        nb_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        frequency_groups = np.array([])
        fmins = np.array([])
        for i in np.arange(nb_clusters):
            ifg = db.labels_ == i
            frequency_groups = np.append(frequency_groups, self.filter(boolean=ifg))
            fmins = np.append(fmins, frequency_groups[i].freqs.min())
        frequency_groups = frequency_groups[np.argsort(fmins)]
        return frequency_groups

    def match(
        self,
        model,
        tolerance=5e-4,
        max_miss=5,
    ):
        """Finds common modes with a reference spectrum.

        Args:
            model (Spectrum): Reference spectrum.
            tolerance (float): Relative period difference tolerated.
            max_miss (int): Maximum number of consecutive missing modes.

        Returns:
            Spectrum: Spectrum containing only the common modes.

        """
        # Keep only the part of model that covers observed periods
        model = model.filter(
            periodmin=(1 - 2 * tolerance) * min(self.periods),
            periodmax=(1 + 2 * tolerance) * max(self.periods),
        )

        i_spectrum = []  # indices of common frequencies in self
        i_model = []  # in model
        #
        if hasattr(self, "amps"):
            for k in np.arange(len(model.periods)):
                i_match = np.argwhere(
                    abs(self.periods / model.periods[k] - 1) <= tolerance
                )[:, 0]
                if i_match.size > 0:
                    i_match = i_match[np.argmax(self.amps[i_match])]
                    i_spectrum.append(i_match)
                    i_model.append(k)
        #
        else:
            for k in np.arange(len(model.periods)):
                i_match = np.argmin(self.periods - model.periods[k])
                if abs(self.periods / model.periods[k] - 1) <= tolerance:
                    i_spectrum.append(i_match)
                    i_model.append(k)

        i_spectrum = i_spectrum = np.array(i_spectrum)
        i_model = np.array(i_model)

        # Remove duplicate matches
        # Several modes in model may be matched with the same mode in self
        # usually because tolerance is too high
        to_delete = np.array([]).astype(int)
        k = 0
        while k < len(i_spectrum):
            j_dupli = np.argwhere(i_spectrum == i_spectrum[k])[:, 0]
            if j_dupli.size > 1:
                diff = abs(
                    self.periods[i_spectrum[j_dupli]] - model.periods[i_model[j_dupli]]
                )
                to_delete = np.append(
                    to_delete, j_dupli[0] + np.argwhere(diff != np.min(diff))[:, 0]
                )
            k += j_dupli.size
        i_spectrum = np.delete(i_spectrum, to_delete)
        i_model = np.delete(i_model, to_delete)

        # max_miss (to be implemented)

        #
        matched = deepcopy(self)
        matched.periods = matched.periods[i_spectrum]
        matched.freqs = matched.freqs[i_spectrum]
        if hasattr(matched, "amps"):
            matched.amps = matched.amps[i_spectrum]
        if hasattr(matched, "n"):
            matched.n = matched.n[i_spectrum]
        if hasattr(matched, "errs"):
            matched.errs = matched.errs[i_spectrum]
        if hasattr(matched, "periods_co"):
            matched.periods_co = matched.periods_co[i_spectrum]

        matched.periods2 = model.periods[i_model]
        matched.freqs2 = model.freqs[i_model]
        if hasattr(model, "amps"):
            matched.amps2 = model.amps[i_model]
        if hasattr(model, "n"):
            matched.n2 = model.n[i_model]
        if hasattr(model, "errs"):
            matched.errs2 = model.errs[i_model]
        if hasattr(model, "periods_co"):
            matched.periods_co2 = model.periods_co[i_model]
        print(len(matched.periods))
        return matched

    def sort(self, by="periods", ascending=True):
        """Sorts modes in spectrum.

        Args:
            by (str): Name of the quantity to sort by.
            ascending: If True, sorts in ascending order else sorts descending.

        Returns:
            Spectrum: Sorted spectrum.
        """
        if by == "periods":
            i_sorted = np.argsort(self.periods)
        elif by == "freqs":
            i_sorted = np.argsort(self.freqs)
        else:
            i_sorted = np.arange(len(self.periods))
        if ascending == False:
            i_sorted = i_sorted[::-1]

        self.periods = self.periods[i_sorted]
        self.freqs = self.freqs[i_sorted]
        if hasattr(self, "amps"):
            self.amps = self.amps[i_sorted]
        if hasattr(self, "n"):
            self.n = self.n[i_sorted]
        if hasattr(self, "errs"):
            self.errs = self.errs[i_sorted]
        if hasattr(self, "periods_co"):
            self.periods_co = self.periods_co[i_sorted]
        if hasattr(self, "periods2"):
            self.periods2 = self.periods2[i_sorted]
            self.freqs2 = self.freqs2[i_sorted]
        if hasattr(self, "amps2"):
            self.amps2 = self.amps2[i_sorted]
        if hasattr(self, "n2"):
            self.n2 = self.n2[i_sorted]
        if hasattr(self, "errs2"):
            self.errs2 = self.errs2[i_sorted]
        if hasattr(self, "periods_co2"):
            self.periods_co2 = self.periods_co2[i_sorted]


def find_zeros(x, y, nb_iterations):
    """Finds the zero of a function y(x) by the secant method. Only works for a
    strictly monotonic function.

    Args:
        x (np.array): Arguments.
        y (np.array): Functions values.
        nb_iterations (int): Number of iterations.

    Returns:
        float: The zero.
    """
    # Finding an upper and a lower bound of the zero
    i_min = np.argmin(abs(y))
    x0 = x[i_min]
    if y[i_min] > 0:
        x1 = x[i_min + 1]
    else:
        x1 = x[i_min - 1]
    # Interpolating the function
    g = interp1d(x, y, kind="quadratic")
    # Root-finding algorithm (secant method)
    for i in np.arange(nb_iterations):
        y_x1 = g(x1)
        y_x0 = g(x0)
        if abs(y_x1) < 1e-14 or abs(y_x0) < 1e-14:
            if y_x1 > y_x0:
                return x0
            else:
                return x1
        x_temp = x1
        x1 = x1 - y_x1 * (x1 - x0) / (y_x1 - y_x0)
        x0 = x_temp
    return x1
