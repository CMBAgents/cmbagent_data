# How to compute and plot CMB Cl's with classy_sz

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


# Compute CMB power spectra
cls = classy_sz.get_cmb_cls(params_values_dict = params)
ell = cls['ell']
cl_tt = cls['tt']
cl_ee = cls['ee']
cl_te = cls['te']

```

## Plot


```python
# Plot the CMB temperature power spectrum
plt.figure(figsize=(10, 5))
plt.plot(ell, ell * (ell + 1) * cl_tt / (2 * np.pi), label='TT')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Multipole moment $\ell$')
plt.ylabel(r'$\ell(\ell+1)C_\ell/(2\pi)$')
plt.title('CMB Temperature Power Spectrum')
plt.legend()
plt.grid(True)
plt.show()

```

    

