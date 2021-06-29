import morse as ms
import matplotlib.pyplot as plt

m = -1
k = 0
nurot = 20
P0 = 4320
factor = 1e6/86400

# Generate a synthetic spectrum
spectrum = ms.Spectrum() 
spectrum.generate(m,k,nurot,P0,nmin=19,nmax=31)



with plt.xkcd():
    fig, [ax1, ax3, ax2] = plt.subplots(1,3,figsize=(9,2),gridspec_kw={'width_ratios':[0.47,0.06,0.47]})
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)    

    # Plot "observed" periods    
    ax1.vlines(spectrum.periods,0,1,colors='k')    
    ax1.set_xlabel(r'$P_{\rm in}$',labelpad=12)
    ax1.set_title('Observed Periods',pad=10,fontsize=14)    
    ax1.set_ylim([0,1.2])    
    
    # Stretch the periods
    eigenvalue = ms.Eigenvalue(m, k)
    Pco = ms.in2co(spectrum.periods,m,k,nurot/factor)
    stretched_periods = ms.stretch(m,k,Pco,eigenvalue,nurot/factor)
    
    # Plot stretched periods
    ax2.vlines(stretched_periods,0,1,colors='k')    
    ax2.set_xlabel(r'$\sqrt{\lambda}P_{\rm co}$',labelpad=10)
    ax2.set_title('Stretched Periods',pad=10,fontsize=14)
    ax2.set_ylim([0,1.2])    
    # Indicate the spacings are equal to P0
    ax2.annotate(text='',xy=(0.995,1.05),xytext=(1.055,1.05),
            fontsize=12,
            arrowprops={'arrowstyle':'<|-|>','facecolor':'k'})
    ax2.text(1.025,1.28,r'$P_0$',fontsize=12,va='center',ha='center')
    
    ax3.set_visible(False)
    
    
plt.show()
fig.savefig('IllusStretchSpectrum.png')


plt.close('all')
import numpy as np

with plt.xkcd():
    fig, ax = plt.subplots(figsize=(4.5,2))
    
    ax.set_xticks([20,40])
    ax.set_yticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)    
    
    
    f0_vect = np.arange(0,3*86400/P0,0.1)
    ax.plot(f0_vect, ms.dft(stretched_periods*np.linspace(0.9,1.1,len(stretched_periods)),f0_vect),c='k')
    
plt.show()
