# How to compute the critical density with classy_sz

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
    'A_s': 2.1e-9,
    'n_s': 0.965,
    'h': 0.67,
    'omega_b': 0.022,
    'omega_cdm': 0.12,
    'tau_reio': 0.06
}


At redshift z, the critical density is computed as:

rho_crit_z = classy_sz_fast.get_rho_crit_at_z(z,params_values_dict=params) # units: $[M_{\odot}/h] [Mpc/h]^{-3}$

# Note:this function handles both scalar and array inputs for z.

```
## Units

The critical density is computed in units of $[M_{\odot}/h] [Mpc/h]^{-3}$.
