#
#
import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.interpolate import interp1d

factor = 1e6 / 86400    # 1 cycle/day in µHz

def co2in(Pco,m,nurot):
    " Compute mode periods in the inertial frame of reference. "
    Pin = Pco / (1 - m*Pco*nurot)
    Pin = abs(Pin)
    return Pin

def in2co(Pin, m, k, nurot, folded=False, ed=False):
    " Compute mode periods in the co-rotating frame of reference. "
    # manual folding
    if folded == False:
        Pco = Pin / (1 + m*Pin*nurot)
    else:
        Pco = - Pin / (1 - m*Pin*nurot)
    # auto folding for quasi-toroidal modes
    if k == -2 and m == 1 and folded == False:
        Pco = - Pin / (1 - m*Pin*nurot)
    if k == -1 and m == 1 and folded == False:
        Pco = - Pin / (1 - m*Pin*nurot)
    # Enlève les modes dont le spin parameter est trop grand ou trop
    # petit à nurot (mais peut garder ceux qui ne peuvent pas exister si m>0)
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
    # Teste s'il reste des modes
    if len(index_keep) == 0:
        print('/!\ No modes left, spin parameters fall outside of tabulated values')
        Pco = np.ones(len(Pin))
    # Gère les sorties
    if ed == False:
        return Pco
    else:
        if len(index_neg) != 0:
            print('/!\ Negative period(s) in the corotating frame at nurotmax = '+\
                        str("%.2f" % (factor*nurot))+' muHz.')
        return Pco, index_keep


def stretch(m, k, periods_co, eigenvalue, nurot):
    """ Stretch the mode periods. """
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
    """ Display a progress bar in terminal."""
    bar_length = 15
    filled_length = int(round(15 * count / total))
    bar = '=' * filled_length + '-' * (bar_length - filled_length)

    percents = 100 * count / total
    sys.stdout.write('[%s]  %s%s %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()
    if count == total: print('\n')
