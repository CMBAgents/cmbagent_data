
# Input parameters

In classy_sz, the baseline input parameters are:

- $ln10^{10}A_s$
- $n_s$
- $H_0$ (in units of km/s/Mpc)
- $\tau_{reio}$
- $\omega_b$ (this is $Omega_b h^2$)
- $\omega_cdm$ (this is $Omega_cdm h^2$)


# Derived parameters

You get the derived parameters by calling the `get_derived_parameters` method (after initialization):

```python
derived_params = classy_sz.get_derived_parameters(cosmo_params)
```


The derived parameters include:
      - "100*theta_s": Scaled angular sound horizon.
      - "sigma8": RMS amplitude of matter fluctuations on 8 Mpc/h scales.
      - "YHe": Helium abundance.
      - "z_reio": Redshift of reionization.
      - "Neff": Effective number of relativistic species.
      - "tau_rec": Conformal time at which the visibility function peaks (recombination time).
      - "z_rec": Redshift at which the visibility function peaks (recombination redshift).
      - "rs_rec": Comoving sound horizon at recombination (in Mpc).
      - "chi_rec": Comoving radial distance to recombination (in Mpc).
      - "tau_star": Conformal time when the photon optical depth equals one.
      - "z_star": Redshift at which the photon optical depth equals one (last scattering surface).
      - "rs_star": Comoving sound horizon at z_star (in Mpc).
      - "chi_star": Comoving radial distance to the last scattering surface (in Mpc).
      - "rs_drag": Comoving sound horizon at the baryon drag epoch (in Mpc).
      - 'h' : dimensionless Hubble parameter h=H0/(100 km/s/Mpc)
      - 'Omega_m'
      - 'Omega_Lambda'
      - 'Omega_r'
      - 'Omega_b'
      - 'Omega_cdm'
      - 'Omega_m_nonu'






