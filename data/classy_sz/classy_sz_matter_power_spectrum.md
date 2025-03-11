# Compute matter power spectrum with classy_sz

We first initialize the computation and then compute.

## Intialize

The initialization has three steps:
    - 1. instantiation of the Class_sz() structure
    - 2. `set` method for requesting outputs.
    - 3. `initialize_classy_szfast` for loading all relevant emulators. 

```python
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from classy_sz import Class as Class_sz
import os
import time

classy_sz = Class_sz()
classy_sz.set({
'output':'mPk', ## ask 'mPk' to specify that we are interested in matter power spectrum.
})
classy_sz.initialize_classy_szfast() ## initialization is crucial! 
```

Once the initialization has been done, it shouldnt be done again!


## Compute linear pk

```python
## pick cosmological parameter values
cosmo_params = {
'omega_b': 0.02242,
'omega_cdm':  0.11933,
'H0': 67.66, # use H0 instead of theta_star because this is what is used by the emulators and to avoid any ambiguity when comparing with camb. 
'tau_reio': 0.0561,
'ln10^{10}A_s': 3.047,
'n_s': 0.9665,
}

z = 0.3 # chose a redshift value 


## compute:
pks,ks = classy_sz.get_pkl_at_z(z,params_values_dict = cosmo_params)
```

## Compute non-linear pk

```python
## pick cosmological parameter values
cosmo_params = {
'omega_b': 0.02242,
'omega_cdm':  0.11933,
'H0': 67.66, # use H0 instead of theta_star because this is what is used by the emulators and to avoid any ambiguity when comparing with camb. 
'tau_reio': 0.0561,
'ln10^{10}A_s': 3.047,
'n_s': 0.9665,
}

z = 0.3 # chose a redshift value 
## compute:
pks,ks = classy_sz.get_pknl_at_z(z,params_values_dict = cosmo_params)
```


`
 
