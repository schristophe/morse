#
#
import importlib.resources

import numpy as np


class Eigenvalue(object):
    """Class to represent an Eigenvalue(m,k) of the Laplace's tidal equation.

    Attributes:
        m (int): Azimuthal order.
        k (int): Ordering index (Lee & Saio 97).
        eta (np.array): Spin factors.
        lamb (np.array): Eigenvalue(m,k) at each spin factor in eta.

    """

    def __init__(self, m, k):
        """Initialises an instance of Eigenvalue.

        Args:
            m (int): Azimuthal order.
            k (int): Ordering index (Lee & Saio 97).

        """
        self.m = m
        self.k = k
        # Loading the tabulated values of Eigenvalue
        if self.k + abs(self.m) > 0:
            if self.m <= 0:
                file_rel_path = f"lambda/lambda_m{self.m}_gyre.txt"
            else:
                file_rel_path = f"lambda/lambda_m{self.m}.txt"
            ref = importlib.resources.files("morse") / file_rel_path
            with importlib.resources.as_file(ref) as file_abs_path:
                try:
                    data = np.genfromtxt(file_abs_path)
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"Tabulated values for lambda(m={self.m},k={self.k}) "
                        + "are not available yet."
                    )
            self.eta = data[:, 0]
            l = self.k + abs(self.m)  # angular degree
            if self.m == 0:
                self.lamb = data[:, l]
            else:
                self.lamb = data[:, l - abs(self.m) + 1]
        elif self.k + abs(self.m) <= 0 and self.m != 0:
            file_rel_path = f"lambda/lambda_m{self.m}_k{self.k}_gyre.txt"
            ref = importlib.resources.files("morse") / file_rel_path
            with importlib.resources.as_file(ref) as file_abs_path:
                try:
                    data = np.genfromtxt(file_abs_path)
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"Tabulated values for lambda(m={self.m},k={self.k}) "
                        + "are not available yet."
                    )
            self.eta = data[:, 0]
            self.lamb = data[:, 1]
        else:
            raise ValueError("Radial modes are not g modes.")
