
import numpy as np

class FAQAState:
    """
    Function-Axis state on a 1D grid representing a wavefunction psi(x).
    """
    def __init__(self, x, psi):
        self.x = x  # 1D grid
        self.dx = x[1] - x[0]
        self.psi = psi.astype(np.complex128)
        self.normalize()

    @classmethod
    def gaussian(cls, x, x0=0.0, sigma=0.5, k0=0.0):
        """Coherent-like state: Gaussian envelope with a plane-wave phase e^{ik0 x}"""
        norm = (1.0/np.sqrt(np.sqrt(2*np.pi)*sigma))
        psi = norm * np.exp(-0.5*((x-x0)/sigma)**2 + 1j*k0*x)
        return cls(x, psi)

    def copy(self):
        return FAQAState(self.x.copy(), self.psi.copy())

    def normalize(self):
        prob = np.sum(np.abs(self.psi)**2)*self.dx
        if prob > 0:
            self.psi /= np.sqrt(prob)

    def prob_density(self):
        return np.abs(self.psi)**2

    def expectation_x(self):
        return np.sum(self.x * self.prob_density()) * self.dx

    def overlap(self, other):
        return np.sum(np.conjugate(self.psi) * other.psi) * self.dx
