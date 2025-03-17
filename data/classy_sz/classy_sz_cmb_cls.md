# How to compute CMB Cl's with classy_sz

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
    'tau_reio': 0.06,
    'omega_b': 0.022,   # this is $Omega_b h^2$
    'omega_cdm': 0.12,  # this is $Omega_cdm h^2$
}


# Compute CMB power spectra
cls = classy_sz.get_cmb_cls(params)
ell = cls['ell']
cl_tt = cls['tt'] # temperature-temperature (TT)
cl_ee = cls['ee'] # polarization-polarization (E mode polarization)
cl_te = cls['te'] # temperature-polarization 
cl_pp = cls['pp'] # lensing potential 'phi-phi'

```

## Units

The CMB power spectra are computed in dimensionless units.


## Plot

Generally, the CMB power spectra should be plotted in log-log scale.

```python
# Plot the CMB temperature power spectrum
plt.figure(figsize=(10, 5))
plt.plot(ell, ell * (ell + 1) * cl_tt / (2 * np.pi), label='TT')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Multipole moment $\ell$')
plt.ylabel(r'$\ell(\ell+1)C_\ell/(2\pi)$') # i.e, also called $D_l$
plt.title('CMB Temperature Power Spectrum')
plt.legend()
plt.grid(True)
plt.show()

```

# Convention for CMB lensing

In classy_sz, the lensing power spectrum is C_l^phi-phi (i.e. lensing potential).

You may want to deal instead with the convergence Cls, then use: C_l^kappa-kappa = [ell(ell+1)/2]**2 * C_l^phi-phi.


To **plot the lensing convergence**, never use D_l, stick to C_l^kappa-kappa. Use log-log scale for both x and y axes, unless otherwise specified.






