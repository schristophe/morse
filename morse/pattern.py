#
#
import numpy as np
import matplotlib.pyplot as plt

class Pattern(object):
    """ Class to represent a period spacing pattern.

        Attributes:

    """

    def __init__(self,spectrum,m,k,nurot,buoyancy_radius,folded,tolerance):
        """ Initialises an instance of Pattern """
        self.m = m
        self.k = k
        self.nurot = nurot
        self.buoyancy_radius = buoyancy_radius
        self.folded = folded
        self.tolerance = tolerance
        # Get the mean offset
        periods_co = in2co(spectrum.periods,m,k,nurot,folded)
        eigenvalue = Eigenvalue(m,k)
        stretched_periods = stretch(periods_co,eigenvalue,nurot)
        stretched_periods_mod = np.mod(stretch_periods,buoyancy_radius)
        offset = np.mean(stretch_periods_mod) / buoyancy_radius
        # Adjust offset if it is close to 0 or 1
        modplus = np.argwhere(stretch_periods_mod < 0.10 * buoyancy_radius)[:,0]
        modminus = np.argwhere(stretch_periods_mod > 0.90 * buoyancy_radius)[:,0]
        if (len(modplus) > 0) & (len(modminus) > 0):
            if len(modplus) > len(modminus):
                stretched_periods_mod = np.where(stretched_periods_mod < 0.50 * buoyancy_radius, stretched_periods_mod + buoyancy_radius, stretched_periods_mod)
                offset = np.mean(stretched_periods_mod) / buoyancy_radius
            else:
                stretched_periods_mod = np.where(stretched_periods_mod > 0.50 * buoyancy_radius, stretched_periods_mod - buoyancy_radius, stretched_periods_mod)
                offset = np.mean(stretched_periods_mod) / buoyancy_radius
        if abs(offset) >= 1:
            print('Warning: the offset was larger than 1.00.')
            offset = offset - 1
        # Generate a synthetic spectrum within the asymptotic traditional approximation
        synth_spectrum = Spectrum().generate(m,k,nurot,buoyancy_radius,offset=offset)
        # Filter synthetic modes that are not in spectrum
        filt_synth_spectrum = synth_spectrum.filter(periodmin=(1 - 5 * tolerance) * min(spectrum.periods),periodmax=(1 + 5 * tolerance) * max(spectrum.periods))
        # For each observed mode, associate the nearest synthetic mode if the period difference is inferior to tolerance
        PaTARf = np.array([])
        Pf = np.array([])
        for i in np.arange(len(P)):
            Ptemp = P[i]*np.ones(len(PaTAR))
            iaTAR = np.argmin(abs(Ptemp - PaTAR))
            if abs(PaTAR[iaTAR]-P[i]) / P[i] <= tolrel:
                PaTARf = np.append(PaTARf,PaTAR[iaTAR])
                Pf = np.append(Pf,P[i])
            else:
                print('No aTAR modes correspond to P = '+"%.3f" % P[i]+' d')
        err_P = np.abs(np.std(Pf-PaTARf))
        #err_P = np.mean(np.std(np.mod(Pp,P0))/np.sqrt(interpolation(Plamb,lamb,PaTARf)))
        return PaTARf,Pf,offset*P0,err_P

    def fit(self):
        """ """

    def plot(self):
        """ """

    def save(self):
        """ """
