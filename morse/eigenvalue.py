#
#
import numpy as np
import sys

class Eigenvalue(object):
    """ Class to represent an Eigenvalue(m,k) of the Laplace's tidal equation.

    Attributes:
        m (int): Azimuthal order.
        k (int): Ordering index (Lee & Saio 97).
        eta (np.array): Spin factors.
        lamb (np.array): Eigenvalue(m,k) at each spin factor in eta.

    """

    def __init__(self,m,k):
        """ Initialises an instance of Eigenvalue.

        Args:
            m (int): Azimuthal order.
            k (int): Ordering index (Lee & Saio 97).

        """
        self.m = m
        self.k = k
        # Loading the tabulated values of Eigenvalue
        if self.k + abs(self.m) > 0:
            try:
                data = np.genfromtxt(f'/home/schristophe/morse/morse/lambda/lambda_m{self.m}.txt')
            except OSError:
                sys.exit(f'Tabulated values for lambda(m={self.m},k={self.k}) '+
                        'are not available yet.')
            self.eta = data[:,0]
            l = self.k + abs(self.m) # angular degree
            if self.m == 0:
                self.lamb = data[:,l]
            else:
                self.lamb = data[:,l-abs(self.m)+1]
        elif self.k + abs(self.m) <= 0 and self.m !=0:
            try:
                data = np.genfromtxt(f'/home/schristophe/morse/morse/lambda/lambda_m{self.m}_k{self.k}.txt')
            except OSError:
                sys.exit(f'''Tabulated values for lambda(m={self.m},k={self.k})
                        are not available yet.''')
            self.eta = data[1:,0]
            self.lamb = data[1:,1] # at s=-k(-k+1), lambda is equal to 0
        else:
            sys.exit('Radial modes are not handled')
