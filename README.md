
# FAQA Prototype (Function-Axis Quantum Architecture)

A minimal, hackathon-ready simulation of function-axis quantum-like computation on a 1D grid.
It demonstrates state preparation, functional "gates" (squeezing, displacement, Fourier, cubic-phase), two-mode mixing, and homodyne-like measurement.

## Quickstart

```bash
pip install numpy matplotlib
python demo.py
```

Artifacts will be written to `/mnt/data` when run here, or to your local directory if cloned locally.

## What this shows
- Infinite-dimensional state encoded on a grid (approximation to L^2(R)).
- Gate pipeline: squeezing -> cubic phase (non-Gaussian) -> displacement -> Fourier.
- Measurement via sampling from |psi(x)|^2.
- Plots: input vs output density, k-space spectrum, measurement histogram.

## Notes
- This is a *simulation* to demonstrate the FAQA concept for D3CODE.
- Physical realizations would map these functional operators to photonic elements (squeezers, modulators, interferometers).
