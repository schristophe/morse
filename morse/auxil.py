#
#
import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.interpolate import interp1d

factor = 1e6 / 86400    # 1 cycle/day in µHz

def co2in(Pco,m,nurot):
    """ Computes mode periods in the inertial frame of reference.

    Args:
        Pco (np.array): Mode periods in the co-rotating frame (in d).
        m (int): Azimuthal order.
        nurot (float): Rotation frequency (in c/d).

    Returns:
        np.array: Mode periods in the inertial frame (in d).

    """
    Pin = Pco / (1 - m*Pco*nurot)
    Pin = abs(Pin)
    return Pin

def in2co(Pin, m, k, nurot, folded=False, ed=False):
    """ Computes mode periods in the co-rotating frame of reference.

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
        Pco = Pin / (1 + m*Pin*nurot)
    else:
        Pco = - Pin / (1 - m*Pin*nurot)
    # Auto-folding for quasi-toroidal modes
    if k == -2 and m == 1 and folded == False:
        Pco = - Pin / (1 - m*Pin*nurot)
    if k == -1 and m == 1 and folded == False:
        Pco = - Pin / (1 - m*Pin*nurot)
    # Discarding modes for which the spin parameter is too large or too small
    # (but may keep those that cannot exist if m < 0)
    index_neg = np.where(Pco < 0)[0]
    s = 2*np.abs(Pco)*nurot # obs. spin parameter
    #print(s)
    if k + abs(m) > 0:
        index_keep = np.where(s <= 20)[0]
    elif k == -2 and m == 1:
        index_keep = np.where((s <= 100) & (s >= -k*(-k-1)+0.05))[0]
    else:
        index_keep = np.where((s <= 19.80) & (s >= -k*(-k-1)+0.05))[0]
    Pco = np.abs(Pco)[index_keep]
    #  Checking if there are any modes left.
    if len(index_keep) == 0:
        print('/!\ No modes left, spin parameters fall outside of tabulated values')
        Pco = np.ones(len(Pin))
    # Managing outputs
    if ed == False:
        return Pco
    else:
        if len(index_neg) != 0:
            print('/!\ Negative period(s) in the corotating frame at nurotmax = '+\
                        str("%.2f" % (factor*nurot))+' muHz.')
        return Pco, index_keep


def stretch(m, k, periods_co, eigenvalue, nurot):
    """ Stretches the mode periods.

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
        interp_lamb = interp1d(P,eigenvalue.lamb,kind='cubic')
        stretched_periods = periods_co * np.sqrt(interp_lamb(periods_co))
    elif nurot == 0:
        if k + abs(m) > 0:
            l = k + abs(m)  # angular degree
            stretched_periods = np.sqrt(l*(l+1)) * periods_co
        else:
            sys.exit('(k = '+str(k)+', m = '+str(m)+') modes does not exist without rotation')
    else:
        sys.exit('Rotation frequency cannot be negative')
    return stretched_periods


def progress_bar(count, total, status=''):
    """ Displays a progress bar in terminal.

    Args:
        count (int): Iteration number.
        total (int): Total number of iterations.
        status (str): Message to display next to the progress bar.

    """
    bar_length = 15
    filled_length = int(round(15 * count / total))
    bar = '=' * filled_length + '-' * (bar_length - filled_length)

    percents = 100 * count / total
    sys.stdout.write('[%s]  %s%s %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()
    if count == total: print('\n')
