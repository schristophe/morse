#
#
import numpy as np
import matplotlib.pyplot as plt
import os

from .eigenvalue import*
from .auxil import *


class EchelleDiagram(object):
    """ Class to represent an echelle diagram.

    Attributes:
        spectrum (Spectrum): Frequency data.
        m (int): Azimuthal order.
        k (int): Ordering index (Lee & Saio 97).
        nurot (float): Rotation frequency (in µHz).
        buoyancy_radius (float): Buoyancy radius (in s).
        folded (bool):
            If True, it is assumed that the spectrum is folded in the inertial
            frame.
        index_keep (np.array):
            Indices of modes in spectrum that were not discarded in the change
            of reference frame.
        stretched_periods_mod (np.array):
            Remainder of the floor division of the stretched periods by the
            buoyancy radius (i.e. modulo).

    """

    def __init__(self,spectrum,m,k,nurot,buoyancy_radius,folded=False):
        """ Initialises an instance of EchelleDiagram.

        The periods are stretched using the parameters (m, k, nurot).

        Args:
            spectrum (Spectrum): Frequency data.
            m (int): Azimuthal order.
            k (int): Ordering index (Lee & Saio 97).
            nurot (float): Rotation frequency (in µHz).
            buoyancy_radius (float): Buoyancy radius (in s).
            folded (bool):
                If True, it is assumed that the spectrum is folded in the
                inertial frame.

        """
        self.spectrum = spectrum
        self.m = m
        self.k = k
        self.nurot = nurot
        self.buoyancy_radius = buoyancy_radius
        self.folded = folded
        eigenvalue = Eigenvalue(m, k)
        periods_co, self.index_keep = in2co(spectrum.periods, m, k, nurot / factor, folded, ed=True)   # moving to the co-rotating frame of reference
        stretched_periods = stretch(m, k, periods_co, eigenvalue, nurot / factor)
        self.stretched_periods_mod = np.mod(stretched_periods, buoyancy_radius / 86400)

    def plot(self, save=False):
        """ Plots the echelle diagram.

        Args:
            save (bool): If True, saves the figure to a png file.

        """
        plt.figure()
        if hasattr(self.spectrum,'amps'):
            max_amp = max(self.spectrum.amps[self.index_keep])
            min_amp = min(self.spectrum.amps[self.index_keep])
            b = (20.0 - 100.0 * (min_amp / max_amp)) / (1.0 - (min_amp / max_amp))
            a = (100.0 - b) / max_amp
            pointsize_ed = a * self.spectrum.amps[self.index_keep] + b
            plt.scatter(86400 * self.stretched_periods_mod, self.spectrum.freqs[self.index_keep], s=pointsize_ed)
            plt.scatter(86400 * self.stretched_periods_mod + self.buoyancy_radius, self.spectrum.freqs[self.index_keep], s=pointsize_ed)
        else:
            plt.scatter(86400 * self.stretched_periods_mod, self.spectrum.freqs[self.index_keep])
            plt.scatter(86400 * self.stretched_periods_mod + self.buoyancy_radius, self.spectrum.freqs[self.index_keep])
        plt.xlabel(r'$\sqrt{\lambda_{k,m}}P_{\rm co}$ mod $P_0 = $'+("%.0f" % (self.buoyancy_radius))+' $s$')
        plt.xlim([0, 2.01 * self.buoyancy_radius])
        plt.ylabel(r'$\nu_{\rm in}$ $(\mu Hz)$')
        if save == True:
            if hasattr(self.spectrum,"path"):
                filename = self.spectrum.path.split("/")[-1]+"_"
            else:
                filename = ""
            if not os.path.isdir(os.getcwd()+'/results/'):
                os.mkdir(os.getcwd()+"/results/")
            filename = os.getcwd()+"/results/"+filename+"ED_m"+str("%d" % self.m)+"_k"+str("%d" % self.k)+".png"
            plt.savefig(filename)
        plt.show()
