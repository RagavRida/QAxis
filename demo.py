import numpy as np
import matplotlib.pyplot as plt
from state import FAQAState
import gates as G
import measure as M
import os

def make_grid(n=2048, x_max=10.0):
    x = np.linspace(-x_max, x_max, n, endpoint=False)
    return x

def pipeline():
    x = make_grid(n=2048, x_max=10.0)
    psi0 = FAQAState.gaussian(x, x0=-2.0, sigma=0.6, k0=4.0)

    # Apply gates: squeeze -> cubic-phase -> displacement -> Fourier
    psi = psi0.copy()
    psi.psi = G.squeezing(psi.psi, psi.x, s=0.8)
    psi.psi = G.cubic_phase(psi.psi, psi.x, gamma=0.01)
    psi.psi = G.displacement(psi.psi, psi.x, a=1.2)
    psi_k = G.fourier(psi.psi)  # transform to k-space

    # Measure <x> before and after
    x_mean_in = psi0.expectation_x()
    x_mean_out = M.expectation(psi.psi, psi.x, op='x')

    # Plot
    fig1 = plt.figure(figsize=(10,4))
    plt.plot(psi0.x, np.abs(psi0.psi)**2, label='|psi_in(x)|^2')
    plt.plot(psi.x, np.abs(psi.psi)**2, label='|psi_out(x)|^2', alpha=0.8)
    plt.legend(); plt.xlabel('x'); plt.ylabel('Probability density'); plt.title('Input vs Output (x-space)')
    plt.tight_layout()
    fig1_path = 'artifacts/faqa_proto_out_x.png'
    os.makedirs(os.path.dirname(fig1_path), exist_ok=True)
    plt.savefig(fig1_path)

    # k-space plot
    k = np.fft.fftshift(np.fft.fftfreq(x.size, d=psi.dx))*2*np.pi
    fig2 = plt.figure(figsize=(10,4))
    plt.plot(k, np.abs(psi_k)**2, label='|psi_out(k)|^2')
    plt.legend(); plt.xlabel('k'); plt.ylabel('Spectral density'); plt.title('Output (k-space)')
    plt.tight_layout()
    fig2_path = 'artifacts/faqa_proto_out_k.png'
    plt.savefig(fig2_path)

    # Homodyne samples
    samples = M.homodyne_x(psi.psi, psi.x, n_samples=10000)
    fig3 = plt.figure(figsize=(10,4))
    plt.hist(samples, bins=100, density=True, alpha=0.7, label='homodyne samples')
    plt.plot(psi.x, np.abs(psi.psi)**2, label='|psi_out(x)|^2')
    plt.legend(); plt.xlabel('x'); plt.ylabel('density'); plt.title('Measurement vs theoretical density')
    plt.tight_layout()
    fig3_path = 'artifacts/faqa_proto_measure.png'
    plt.savefig(fig3_path)

    return {
        'x_mean_in': float(x_mean_in),
        'x_mean_out': float(x_mean_out),
        'fig_x': fig1_path,
        'fig_k': fig2_path,
        'fig_meas': fig3_path
    }

if __name__ == "__main__":
    out = pipeline()
    print(out)
