import morse as ms
import numpy as np
import matplotlib.pyplot as plt

m = -1
k = 0
P0 = 4320
nurot = 20

# Generate a synthetic spectrum
spectrum = ms.Spectrum()
spectrum.generate(m,k,nurot,P0,nmin=19,nmax=30)

# Parameter space to explore
nurot_vect = np.linspace(15,24,500)
f0_vect = np.arange(100,600,0.1)

# Compute IDMap
idmap = ms.IDMap(spectrum,m,k,nurot_vect,f0_vect)

# Plot IDMap
idmap.plot()
plt.plot(1e6/idmap.buoyancy_radius,idmap.nurot,'o',c='tab:red',mec='k') # Max of PSD
plt.savefig('IllusIDMap.png')
