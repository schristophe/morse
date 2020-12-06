#
#
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy

from .auxil import *
from .echellediagram import *
from .eigenvalue import *
from .spectrum import *

class IDMap(object):
    """ """

    def __init__(self, spectrum, m, k, nurot_vect, f0_vect, folded=False):
        """ Initialise an instance of IDMap """
        self.spectrum = spectrum
        self.m = m
        self.k = k
        self.nurot_vect = nurot_vect
        self.f0_vect = f0_vect
        self.folded = folded
        self.eigenvalue = Eigenvalue(m,k)
        # Initialise the ID map and the array that will store the resolution
        # of each DFT spectrum
        resolution = np.nan * np.ones(len(nurot_vect))
        dft_map = np.nan * np.ones((len(nurot_vect),len(f0_vect)))
        # For each value of nurot, stretch the oscillation spectrum and
        # compute its DFT
        nurot_vect = nurot_vect / factor    # convert to c/d for computations
        for i in np.arange(len(nurot_vect)):
            progress_bar(i+1, len(nurot_vect), 'Computing the ID map...')
            nurot = nurot_vect[i]
            periods_co = in2co(spectrum.periods,m,k,nurot,folded)
            if len(periods_co) > 3:
                stretched_periods = stretch(m, k, periods_co,self.eigenvalue,nurot)
                resolution[i] = 1.0 / (np.max(stretched_periods) - np.min(stretched_periods))
                # Compute the DFT of the stretched spectrum
                dft_map[i,:] = dft(stretched_periods, f0_vect / factor) # f0_vect has to be in c/d
        self.resolution = resolution * factor
        self.dft_map = dft_map
        # Looking for which parameters (nurot,f0) maximise the PSD in dft_map
        self.psd_max = np.nanmax(self.dft_map)
        imax_nurot, imax_f0 = np.argwhere(self.dft_map == self.psd_max)[0]
        self.nurot = self.nurot_vect[imax_nurot]
        self.buoyancy_radius = 1e6 / f0_vect[imax_f0] # we want the buoyancy radius (dim of a time)
        # Compute the detection threshold
        # (above which we consider we have detected a period spacing pattern)
        nb_bins = np.ceil((self.f0_vect.max()-self.f0_vect.min())/self.resolution[imax_nurot])
        self.psd_threshold = - np.log(1 - 0.99**(1/nb_bins))
        if self.psd_max > self.psd_threshold:
            self.flag_detection = True
        else:
            self.flag_detection = False

    def get_echelle_diagram(self):
        """ Compute the echelle diagram using the parameters that maximise the PSD in the DFT map. """
        self.echelle_diagram = EchelleDiagram(self.spectrum,self.m,self.k,self.nurot,self.buoyancy_radius,folded=self.folded)

    def get_pattern(self,tolerance):
        """ """

    def get_param_uncertainties(self, ndraws=500, propagate=None, tolerance=5e-3):
        """ Estimate the uncertainty on the rotation frequency and buoyancy radius using Monte-Carlo simulations. """
        if propagate == 'err' and hasattr(self.spectrum,'errs'):        # propagate errors on mode periods
            period_errs = self.spectrum.errs / self.spectrum.errs**2
        else:                                                           # or the mean difference between TAR model and observed modes
            self.tolerance = tolerance
            # Get the mean offset
            periods_co = in2co(self.spectrum.periods, self.m, self.k, self.nurot, self.folded)
            stretched_periods = stretch(self.m,self.k,periods_co, self.eigenvalue, self.nurot)
            stretched_periods_mod = np.mod(stretched_periods, self.buoyancy_radius)
            offset = np.mean(stretched_periods_mod) / self.buoyancy_radius
            # Adjust offset if it is close to 0 or 1
            modplus = np.argwhere(stretched_periods_mod < 0.10 * self.buoyancy_radius)[:,0]
            modminus = np.argwhere(stretched_periods_mod > 0.90 * self.buoyancy_radius)[:,0]
            if (len(modplus) > 0) & (len(modminus) > 0):
                if len(modplus) > len(modminus):
                    stretched_periods_mod = np.where(stretched_periods_mod < 0.50 * self.buoyancy_radius, stretched_periods_mod + self.buoyancy_radius, stretched_periods_mod)
                    offset = np.mean(stretched_periods_mod) / self.buoyancy_radius
                else:
                    stretched_periods_mod = np.where(stretched_periods_mod > 0.50 * self.buoyancy_radius, stretched_periods_mod - self.buoyancy_radius, stretched_periods_mod)
                    offset = np.mean(stretched_periods_mod) / self.buoyancy_radius
            if abs(offset) >= 1:
                print('Warning: the offset was larger than 1.00.')
                offset = offset - 1
            self.offset = offset
            # Generate a synthetic spectrum within the asymptotic traditional approximation
            synth_spectrum = Spectrum()
            synth_spectrum.generate(self.m ,self.k, self.nurot, self.buoyancy_radius, offset=offset)
            # Filter synthetic modes that are not in spectrum
            filt_synth_spectrum = synth_spectrum.filter(periodmin=(1 - 5 * tolerance) * min(self.spectrum.periods),periodmax=(1 + 5 * tolerance) * max(self.spectrum.periods))
            # For each observed mode, associate the nearest synthetic mode if the period difference is inferior to tolerance
            synth_periods = np.array([])
            matched_obs_periods = np.array([])
            filter_obs_spectrum = np.array([])
            for i in np.arange(len(self.spectrum.periods)):
                i_closest = np.argmin(abs(filt_synth_spectrum.periods - self.spectrum.periods[i]))
                if abs(filt_synth_spectrum.periods[i_closest]- self.spectrum.periods[i]) / self.spectrum.periods[i] <= tolerance:
                    synth_periods = np.append(synth_periods, filt_synth_spectrum.periods[i_closest])
                    filter_obs_spectrum = np.append(filter_obs_spectrum,True)
                else:
                    filter_obs_spectrum = np.append(filter_obs_spectrum,False)
                    print('No synthetic mode corresponds to observed period = '+str("%.3f" % self.spectrum.periods[i])+' d')
            matched_obs_spectrum = self.spectrum.filter(boolean = filter_obs_spectrum)
            self.err_periods_mc = np.std(matched_obs_spectrum.periods - synth_periods)
        # Adjust the resolution and bounds of the DFT map so the computing time stays reasonable
        i_bounds = np.argwhere(self.dft_map >= 0.40 * self.psd_max)
        nurot_vect_dg = np.linspace(self.nurot_vect[i_bounds[:,0]].min(),self.nurot_vect[i_bounds[:,0]].max(),100)
        f0_vect_dg = np.linspace(self.f0_vect[i_bounds[:,1]].min(),self.f0_vect[i_bounds[:,1]].max(),100)
        results = np.zeros((ndraws,4))
        for i in np.arange(ndraws):
            progress_bar(i+1, ndraws, status='Evaluating uncertainty on estimated parameters...')
            # Draw a spectrum by perturbing mode periods/frequencies
            perturbed_spectrum = deepcopy(matched_obs_spectrum)
            perturbed_spectrum.periods = np.random.normal(loc=perturbed_spectrum.periods, scale=self.err_periods_mc)
            # Compute the DFT map for the perturbed spectrum
            perturbed_idmap = IDMap(perturbed_spectrum, self.m, self.k, nurot_vect_dg, f0_vect_dg, self.folded)
            # Store results
            results[i,:] = np.array([perturbed_idmap.nurot,
                    perturbed_idmap.buoyancy_radius,
                    perturbed_idmap.psd_max,
                    perturbed_idmap.flag_detection])
        self.results_mc = results
        self.err_nurot = np.std(results[:,0])
        self.err_buoyancy_radius = np.std(results[:,1])

    def plot(self,cmap='cividis',save=False):
        """ Plot the computed DFT map. """
        plt.figure()
        plt.tick_params(axis='both',direction='inout',which='major',top=False,right=False)
        plt.tick_params(axis='both',direction='inout',which='minor',top=False,right=False)
        plt.pcolormesh(self.f0_vect,self.nurot_vect,self.dft_map,cmap=cmap,shading='nearest',rasterized=True)
        plt.xlabel(r'$1/P_0$ ${\rm (\mu Hz)}$')
        plt.ylabel(r'$\nu_{\rm rot}$ ${\rm (\mu Hz)}$')
        plt.xlim([self.f0_vect.min(), self.f0_vect.max()])
        plt.ylim([self.nurot_vect.min(), self.nurot_vect.max()])
        cb = plt.colorbar(fraction=0.045, pad=0.04)
        cb.set_label(r'$|DFT(\sqrt{\lambda_{m,k}}P_{\rm co})|^2$')
        # Plot contours
        niveau = np.nanmax(self.dft_map) * np.array([0.50,0.95])
        contours = [(0,(10,5)),'solid']
        plt.contour(self.f0_vect,self.nurot_vect,self.dft_map,levels=niveau,linestyles=contours,colors='k')
        if save == True:
            if hasattr(self.spectrum,"path"):
                filename = self.spectrum.path.split("/")[-1]+"_"
            else:
                filename = ""
            if not os.path.isdir(os.getcwd()+'/results/'):
                os.mkdir(os.getcwd()+"/results/")
            filename = os.getcwd()+"/results/"+filename+"IDMAP_m"+str("%d" % self.m)+"_k"+str("%d" % self.k)+".png"
            plt.savefig(filename)
        plt.show()

    def save(self):
        """ Save results, log and plots. """
        if hasattr(self.spectrum,"path"):
            filename = self.spectrum.path.split("/")[-1]+"_"
        else:
            filename = ""
        if not os.path.isdir(os.getcwd()+'/results/'):
            os.mkdir(os.getcwd()+"/results/")
        # Save results
        fname = os.getcwd()+"/results/"+filename+"IDMAP-RESULTS_m"+str("%d" % self.m)+"_k"+str("%d" % self.k)+".txt"
        with open(fname,"w") as fres:
            if hasattr(self,"err_nurot"):
                fres.write(f"{'detection':10s}\t{self.flag_detection}\n" +\
                        f"{'m':10s}\t{self.m:10d}\n" +
                        f"{'k':10s}\t{self.k:10d}\n" +
                        f"{'nurot':10s}\t{self.nurot:10.4f}\n".format("nurot",self.nurot) +
                        f"{'err_nurot':10s}\t{self.err_nurot:10.4f}\n" +
                        f"{'P0':10s}\t{self.buoyancy_radius:10.4f}\n" +
                        f"{'err_P0':10s}\t{self.err_buoyancy_radius:10.4f}\n" +
                        f"{'max_PSD':10s}\t{self.psd_max:10.2f}\n")
            else:
                fres.write(f"{'detection':10s}\t{self.flag_detection}\n" +
                        f"{'m':10s}\t{self.m:10d}\n" +
                        f"{'k':10s}\t{self.k:10d}\n" +
                        f"{'nurot':10s}\t{self.nurot:10.4f}\n" +
                        f"{'P0':10s}\t{self.bouyancy_radius:10.4f}\n" +
                        f"{'max_PSD':10s}\t{self.psd_max:10.2f}\n")
        # Save results of the MC simulation (if applicable)
        if hasattr(self,"err_nurot"):
            fname = os.getcwd()+"/results/"+filename+"IDMAP-RESULTS-MC_m"+str("%d" % self.m)+"_k"+str("%d" % self.k)+".txt"
            hdr = ("{:10s}\t"*4).format("nurot","P0","max_PSD","detection")
            format = "%10.4f "*3+"%10d"
            np.savetxt(fname, self.results_mc, header=hdr, fmt=format)
        # Save log (everything to reproduce the results)
        fname = os.getcwd()+"/results/"+filename+"IDMAP-LOG_m"+str("%d" % self.m)+"_k"+str("%d" % self.k)+".txt"
        with open(fname,"w") as flog:
            flog.write("# SPECTRUM\n")
            if hasattr(self.spectrum,"amps"):
                flog.write(f"ampmin = {min(self.spectrum.amps):8g}\n" +
                    f"ampmax = {max(self.spectrum.amps):8g}\n" +
                    f"freqmin = {min(self.spectrum.freqs):8g}\n" +
                    f"freqmax = {max(self.spectrum.freqs):8g}\n")
            flog.write("\n# IDMAP\n" +
                    f"m = {self.m:d}\n" +
                    f"k = {self.k:d}\n" +
                    f"scan_nurot = ({min(self.nurot_vect):g},{max(self.nurot_vect):g},{len(self.nurot_vect):d})\n" +
                    f"scan_f0 = ({min(self.f0_vect):g},{max(self.f0_vect):g},{len(self.f0_vect):d})\n" +
                    f"folded = {self.folded}\n")
            if hasattr(self,"err_nurot") and hasattr(self,"offset"):
                flog.write("\n# IDMAP MC SIMULATION\n" +
                        "propagate = model_err\n" +
                        f"ndraws = {len(self.results_mc):d}\n"+
                        f"tolerance = {self.tolerance:8g}\n" +
                        f"offset = {self.offset:8g}\n" )
            elif hasattr(self,"err_nurot"):
                flog.write("\n# IDMAP MC SIMULATION\n" +
                        "propagate = obs_err\n" +
                        f"ndraws = {len(self.results_mc):d}\n")
        # Save plots
        self.plot(save=True)
        if not hasattr(self,"echelle_diagram"):
            self.get_echelle_diagram()
        self.echelle_diagram.plot(save=True)
        plt.close('all')

# Auxiliary functions used by IDmap methods

def dft(P,f0_vect):
    """
    Compute the Discrete Fourier Transform (DFT) of P at each frequency
    in f0_vect.
    """
    N = len(P)
    cos = np.zeros(len(f0_vect))
    sin = np.zeros(len(f0_vect))
    for i in np.arange(N):
        for j in np.arange(len(f0_vect)):
            cos[j] = cos[j] + np.cos(2*np.pi*f0_vect[j]*P[i])
            sin[j] = sin[j] + np.sin(2*np.pi*f0_vect[j]*P[i])
    dft = (1.0/N)*(cos**2 + sin**2)
    return dft
