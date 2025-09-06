
import numpy as np

def homodyne_x(psi, x, n_samples=1):
    """Sample measurement outcomes from |psi(x)|^2 using inverse transform sampling."""
    prob = np.abs(psi)**2
    prob /= np.trapz(prob, x)
    cdf = np.cumsum(prob)
    cdf /= cdf[-1]
    u = np.random.rand(n_samples)
    # map uniform samples to x via cdf^{-1}
    idx = np.searchsorted(cdf, u, side='right')
    idx = np.clip(idx, 0, len(x)-1)
    return x[idx]

def expectation(psi, x, op='x'):
    if op == 'x':
        return np.trapz(x * np.abs(psi)**2, x)
    elif op == 'x2':
        return np.trapz((x**2) * np.abs(psi)**2, x)
    else:
        raise ValueError('Unknown operator')
