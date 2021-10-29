#
#
import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.interpolate import interp1d
from scipy.signal import correlate, correlation_lags

from .eigenvalue import *
from .spectrum import *


FACTOR_ROT = 1e6 / 86400  # 1 cycle/day in µHz


def co2in(Pco, m, nurot):
    """Computes mode periods in the inertial frame of reference.

    Args:
        Pco (np.array): Mode periods in the co-rotating frame (in d).
        m (int): Azimuthal order.
        nurot (float): Rotation frequency (in c/d).

    Returns:
        np.array: Mode periods in the inertial frame (in d).

    """
    Pin = Pco / (1 - m * Pco * nurot)
    Pin = abs(Pin)
    return Pin


def in2co(Pin, m, k, nurot, folded=False, ed=False):
    """Computes mode periods in the co-rotating frame of reference.

    Args:
        Pin (np.array): Mode periods in the inertial frame (in d).
        m (int): Azimuthal order.
        k (int): Ordering index (Lee & Saio 97).
        nurot (float): Rotation frequency (in c/d).
        folded (bool):
            If True, it is assumed that the spectrum is folded in the
            inertial frame.
        ed (bool):
            If True, the method also returns the indices of elements in Pin that
            were not discarded in the change of reference frame. This is only
            used by the EchelleDiagram class.

    Returns:
        np.array: Mode periods in the co-rotating frame (in d).
    """
    # Manual folding
    if folded == False:
        Pco = Pin / (1 + m * Pin * nurot)
    else:
        Pco = -Pin / (1 - m * Pin * nurot)
    # Auto-folding for quasi-toroidal modes
    if k == -2 and m == 1 and folded == False:
        Pco = -Pin / (1 - m * Pin * nurot)
    if k == -1 and m == 1 and folded == False:
        Pco = -Pin / (1 - m * Pin * nurot)
    # Discarding modes for which the spin parameter is too large or too small
    # (but may keep those that cannot exist if m < 0)
    index_neg = np.where(Pco < 0)[0]
    s = 2 * np.abs(Pco) * nurot  # obs. spin parameter
    # print(s)
    if k + abs(m) > 0:
        if m >= 0:
            index_keep = np.where(s < 100)[0]
        else:
            index_keep = np.where(s < 20)[0]
    elif k == -2 and m == 1:
        index_keep = np.where((s < 100) & (s >= -k * (-k - 1) + 2e-10))[0]
    else:
        index_keep = np.where((s < 100) & (s >= -k * (-k - 1) + 2e-10))[0]
    Pco = np.abs(Pco)[index_keep]
    #  Checking if there are any modes left.
    if len(index_keep) == 0:
        print("/!\ No modes left, spin parameters fall outside of tabulated values")
        Pco = np.ones(len(Pin))
    # Managing outputs
    if ed == False:
        return Pco
    else:
        if len(index_neg) != 0:
            print(
                "/!\ Negative period(s) in the corotating frame at nurotmax = "
                + str("%.2f" % (FACTOR_ROT * nurot))
                + " muHz."
            )
        return Pco, index_keep


def stretch(m, k, periods_co, eigenvalue, nurot):
    """Stretches the mode periods.

    Args:
        m (int): Azimuthal order.
        k (int): Ordering index (Lee & Saio 97).
        periods_co (np.array): Mode periods in the co-rotating frame (in d).
        eigenvalue (Eigenvalue): Eigenvalue(m,k) of Laplace's tidal equation.
        nurot (float): Rotation frequency (in c/d).

    Returns:
        np.array: The stretched periods (in d).

    """
    if nurot > 0:
        P = eigenvalue.eta / (2 * nurot)
        interp_lamb = interp1d(P, eigenvalue.lamb, kind="cubic")
        stretched_periods = periods_co * np.sqrt(interp_lamb(periods_co))
    elif nurot == 0:
        if k + abs(m) > 0:
            l = k + abs(m)  # angular degree
            stretched_periods = np.sqrt(l * (l + 1)) * periods_co
        else:
            sys.exit(
                "(k = "
                + str(k)
                + ", m = "
                + str(m)
                + ") modes does not exist without rotation"
            )
    else:
        sys.exit("Rotation frequency cannot be negative")
    return stretched_periods


def progress_bar(count, total, status=""):
    """Displays a progress bar in terminal.

    Args:
        count (int): Iteration number.
        total (int): Total number of iterations.
        status (str): Message to display next to the progress bar.

    """
    bar_length = 15
    filled_length = int(round(15 * count / total))
    bar = "=" * filled_length + "-" * (bar_length - filled_length)

    percents = 100 * count / total
    sys.stdout.write("[%s]  %s%s %s\r" % (bar, percents, "%", status))
    sys.stdout.flush()
    if count == total:
        print("\n")


def generate_spectrum(periods, sigma, period_max, sampling_rate):
    """Generates a spectrum by modelling each peak by a Gaussian
    function with a standard deviation of sigma.

    Args:
        periods (np.array): Mode periods.
        sigma (float): Standard deviation of Gaussian functions.
        period_max (float): Returned spectrum is computed up to this period.
        sampling_rate (float): Sampling rate of the return spectrum.


    Returns:
        np.array: Artificially generated spectrum.

    """
    P = np.arange(0, period_max, 1 / sampling_rate)
    spectrum = 0
    for period in periods:
        spectrum = spectrum + np.exp(-((P - period) ** 2) / (2 * sigma ** 2))
    return spectrum


def get_offset(spectrum, m, k, nurot, buoy_r, folded=False):
    """Determines the value of the offset by cross-correlating the stretched
    observed spectrum with the stretched TAR model computed for the parameters
    (m, k, nurot, buoyancy radius).

    Args:
        spectrum (Spectrum):
        m (int): Azimuthal order.
        k (int): Ordering index (Lee & Saio 97).
        nurot (float): Rotation frequency (in c/d).
        buoy_r (float): Buoyancy radius (in d).
        folded (bool):
            If True, it is assumed that the spectrum is folded in the
            inertial frame.

    Returns:
        float: Offset (in d).
    """
    # Observed spectrum: switch to corotating frame and stretch the spectrum
    periods_co = in2co(spectrum.periods, m, k, nurot, folded)
    obs_stretched = stretch(m, k, periods_co, Eigenvalue(m, k), nurot)

    # Generate spectrum model
    spectrum_model = Spectrum()
    spectrum_model.generate(
        m, k, nurot * FACTOR_ROT, buoy_r * 86400, offset=0, nmax=110
    )
    spectrum_model = spectrum_model.filter(periodmax=np.max(spectrum.periods) + 0.5)
    # Stretch it
    mod_stretched = stretch(m, k, spectrum_model.periods_co, Eigenvalue(m, k), nurot)

    # Artifically broaden peaks in the spectra for the cross-correlation
    # (because the generated TAR model is not perfectly modelling the observed
    # periods)
    period_max = np.max(obs_stretched) + 0.5
    sigma = 1e-3
    sampling_rate = 3e4
    obs_broaden = generate_spectrum(obs_stretched, sigma, period_max, sampling_rate)
    mod_broaden = generate_spectrum(mod_stretched, sigma, period_max, sampling_rate)

    # Compute the cross-correlation of the two broaden spectra
    correlation = correlate(obs_broaden, mod_broaden, mode="full")
    lags = correlation_lags(obs_broaden.size, mod_broaden.size, mode="full") / (
        sampling_rate * buoy_r
    )

    # Offset is between 0 and 1 (by definition)
    lag0 = len(lags) // 2
    lag_P0 = lag0 + int(buoy_r * sampling_rate)
    offset = lags[lag0:lag_P0][np.argmax(correlation[lag0:lag_P0])]
    return offset
