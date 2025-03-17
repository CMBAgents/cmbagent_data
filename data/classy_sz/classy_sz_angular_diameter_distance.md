# How to compute the angular diameter distance with classy_sz

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
    'H0': 67.66,
    'omega_b': 0.022, # this is $Omega_b h^2$
    'omega_cdm': 0.12, # this is $Omega_cdm h^2$
    'tau_reio': 0.06
}


# At redshift z, the angular diameter distance is computed as:

angular_distance_z = classy_sz.get_angular_distance_at_z(z,params) # units: $[Mpc]$
# Note:this function handles both scalar and array inputs for z.

```
## Units

The angular diameter distance is computed in units of $[Mpc]$.


## Note

The comoving distance $\chi$ is related to the angular diameter distance $D_A$ at redshift $z$ as:


$$
\chi = D_A * (1+z)
$$


