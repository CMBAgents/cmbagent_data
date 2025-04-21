# Compute matter power spectrum with classy_sz

We first initialize the computation and then compute.

## Intialize

The initialization has three steps:
    - 1. instantiation of the Class_sz() structure
    - 2. `initialize_classy_szfast` for loading all relevant emulators. 


```python
from classy_sz import Class as Class_sz


classy_sz = Class_sz()
classy_sz.initialize_classy_szfast() ## initialization is crucial! 
```

Once the initialization has been done, it shouldnt be done again!


## Compute linear pk

```python
## pick cosmological parameter values
cosmo_params = {
'omega_b': 0.02242, # this is $Omega_b h^2$
'omega_cdm':  0.11933, # this is $Omega_cdm h^2$
'H0': 67.66, # in km/s/Mpc
'tau_reio': 0.0561, # this parameter doesnt matter for P(k)
'ln10^{10}A_s': 3.047,
'n_s': 0.9665,
}

z = 0.3 # chose a redshift value 


## compute:
pks,ks = classy_sz.get_pkl_at_z(z,cosmo_params) # units: $k [1/Mpc]$ and $P(k) [Mpc]^3$
```

## Compute non-linear pk

```python
## pick cosmological parameter values
cosmo_params = {
'omega_b': 0.02242, # this is $Omega_b h^2$
'omega_cdm':  0.11933, # this is $Omega_cdm h^2$
'H0': 67.66, # use H0 instead of theta_star because this is what is used by the emulators and to avoid any ambiguity when comparing with camb. 
'tau_reio': 0.0561, # this parameter doesnt matter for P(k)
'ln10^{10}A_s': 3.047,
'n_s': 0.9665,
}

z = 0.3 # chose a redshift value 
## compute:
pks,ks = classy_sz.get_pknl_at_z(z,cosmo_params) # units: $k [1/Mpc]$ and $P(k) [Mpc]^3$
```

## Range of wavenumbers

The maximum wavenumber is kmax=10 [1/Mpc].

The minimum wavenumber is kmin=1e-4 [1/Mpc].


## Plotting the matter power spectrum

For plots, the matter power spectrum should be presented in log-log scale.

Any function of wavenumber should in general be plotted in log scale for x-axis.


## Interpolating the matter power spectrum

Interpolation on matter power spectrum is done in log-log space.

