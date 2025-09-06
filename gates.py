
import numpy as np

def displacement(psi, x, a):
    """Shift psi(x) -> psi(x - a). Implement via interpolation on grid."""
    # simple linear interpolation; assume uniform grid
    dx = x[1]-x[0]
    shift = int(np.round(a/dx))
    return np.roll(psi, shift)

def squeezing(psi, x, s):
    """Scaling (squeezing) x -> s x, with Jacobian compensation to keep L2 norm."""
    # Resample using linear interpolation
    x_prime = x / s
    psi_interp = np.interp(x_prime, x, psi.real, left=0.0, right=0.0) + 1j*np.interp(x_prime, x, psi.imag, left=0.0, right=0.0)
    # Jacobian factor |s|^{1/2} to preserve normalization in continuous case
    return np.sqrt(np.abs(s)) * psi_interp

def fourier(psi):
    """Unitary Fourier transform on finite grid (using FFT with unitary normalization)."""
    n = psi.size
    out = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(psi)))/np.sqrt(n)
    return out

def ifourier(psi_k):
    n = psi_k.size
    out = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(psi_k)))*np.sqrt(n)
    return out

def cubic_phase(psi, x, gamma):
    """Non-Gaussian gate U = exp(i gamma x^3)."""
    return psi * np.exp(1j*gamma*(x**3))

def phase(psi, phi):
    return psi * np.exp(1j*phi)

def two_mode_mix(psi1, psi2, theta):
    """Beam-splitter-like mixing for two independent modes (same grid)."""
    # simple SU(2) rotation in mode space
    a = np.cos(theta)*psi1 + np.sin(theta)*psi2
    b = -np.sin(theta)*psi1 + np.cos(theta)*psi2
    return a, b
