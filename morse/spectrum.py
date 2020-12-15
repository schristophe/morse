#
#
import numpy as np
import matplotlib.pyplot as plt
#import astropy.units as u
from scipy.interpolate import interp1d
from copy import deepcopy

from .auxil import *
from .eigenvalue import *

class Spectrum(object):
    """ Class to represent frequency data either extracted from the
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
        """ Initialises an instance of Spectrum """

    def load(self,path='test/kic8375138_test.freq', colfreqs=0, colerrs=-1,
        ufreqs='u.microHertz', colamps=-1):
        """ Loads frequency data from a file.

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
        self.freqs = data[:,colfreqs]  #* ufreqs
        self.periods = factor / self.freqs
        if colerrs != -1:
            self.errs = data[:,colerrs]  #* ufreqs
        if colamps != -1:
            self.amps = data[:,colamps]

    def generate(self,m,k,nurot,buoyancy_radius,offset=0.0,nmin=1,nmax=90):
        """ Generates synthetic frequency data using the asymptotic formulation
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
            n = np.arange(nmin,nmax+1,1)
        elif k + abs(m) <= 0:
            n = np.arange(nmin,nmax+1,1)
        self.n = n
        nurot = nurot / factor # nurot has to be in c/d
        buoyancy_radius = buoyancy_radius / 86400. # buoyancy_radius in d
        if nurot > 0:
            eigenvalue = Eigenvalue(m,k)
            P = eigenvalue.eta / (2*nurot)
            f_P_k = buoyancy_radius / np.sqrt(eigenvalue.lamb)
            periods_co = np.array([])
            for i in n:
                f_P = f_P_k * (i + offset)
                periods_co = np.append(periods_co,find_zeros(P,f_P - P,9))
        else:
            if k + abs(m) > 0:
                l = k + abs(m)
                periods_co = buoyancy_radius * (n + offset) / np.sqrt(l * (l+1))
            elif k + abs(m) <= 0 and m != 0:
                sys.exit('(k = '+str(k)+', m = '+str(m)+') modes does not exist without rotation.')
            else:
                sys.exit('Radial modes are not handled in this version of the code.')
        self.periods_co = periods_co
        self.periods = co2in(periods_co,m,nurot)
        self.freqs = factor / self.periods  # in µHz


    def plot(self):
        """ Plots the oscillation spectrum. """
        plt.figure()
        plt.vlines(self.freqs,0,self.amps,lw=1)
        plt.xlabel('Frequency ('+str(self.ufreqs)+')')
        plt.ylabel('Amplitude')
        plt.show()

    def search_combinations(self, order=2):
        """ """

    def filter(self,ampmin=0,ampmax=np.inf,freqmin=0,freqmax=np.inf,periodmin=0,periodmax=np.inf,nmin=0,nmax=np.inf,boolean=None):
        """ Filters the spectrum according to the period/frequency/amplitude of
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
        if hasattr(self,'amps'):
            conditions = np.vstack((conditions,np.array([self.amps >= ampmin, self.amps <= ampmax])))
        if hasattr(self,'n'):
            conditions = np.vstack((conditions,np.array([self.n >= nmin, self.n <= nmax])))
        conditions = np.vstack((conditions,np.array([self.freqs >= freqmin, self.freqs <= freqmax, self.periods >= periodmin, self.periods <= periodmax])))
        filter = np.logical_and.reduce(conditions)
        # Creating a new Spectrum object with modes that pass the filter
        filtered_spectrum = deepcopy(self)
        filtered_spectrum.periods = filtered_spectrum.periods[filter]
        filtered_spectrum.freqs = filtered_spectrum.freqs[filter]
        if hasattr(filtered_spectrum,'amps'):
            filtered_spectrum.amps = filtered_spectrum.amps[filter]
        if hasattr(filtered_spectrum,'n'):
            filtered_spectrum.n = filtered_spectrum.n[filter]
        if hasattr(filtered_spectrum,'errs'):
            filtered_spectrum.errs = filtered_spectrum.errs[filter]
        if hasattr(filtered_spectrum,'periods_co'):
            filtered_spectrum.periods_co = filtered_spectrum.periods_co[filter]
        return filtered_spectrum

    def clustering(self):
        """ """


def find_zeros(x,y,nb_iterations):
    """ Finds the zero of a function y(x) by the secant method. Only works for a
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
    g = interp1d(x,y,kind='quadratic')
    # Root-finding algorithm (secant method)
    for i in np.arange(nb_iterations):
        y_x1 = g(x1)
        y_x0 = g(x0)
        if abs(y_x1) < 1e-8 or abs(y_x0) < 1e-8:
            if y_x1 > y_x0:
                return x0
            else:
                return x1
        x_temp = x1
        x1 = x1 - y_x1 * (x1 - x0)  / (y_x1 - y_x0)
        x0 = x_temp
    return x1
