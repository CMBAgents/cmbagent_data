# How to compute the Hubble parameter with classy_sz


## Intialize

```python
from classy_sz import Class as Class_sz
classy_sz = Class_sz()
classy_sz.initialize_classy_szfast() ## initialization is crucial! 
```


## Compute

```python
# Set cosmological parameters
params = {
    'ln10^{10}A_s': 3.047,
    'n_s': 0.965,
    'H0': 67.66, # in km/s/Mpc
    'omega_b': 0.022,   # this is $Omega_b h^2$
    'omega_cdm': 0.12,  # this is $Omega_cdm h^2$
    'tau_reio': 0.06
}


# At redshift z, the Hubble parameter (in units of $1/Mpc$) is computed as:

H_z = classy_sz.get_hubble_at_z(z,params) # units: [1/Mpc]
# Note:this function handles both scalar and array inputs for z.

```
## Units

The Hubble parameter is computed in units of $[1/Mpc]$.






