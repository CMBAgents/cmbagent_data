# Classy_sz Initialization and Computing Guide

The file is structured into sections to make it **RAG-friendly** and **easy to parse**. Each section is self-contained so a retrieval agent can extract the relevant context without losing important details.

For examples of codes in classy_sz, see [See Classy_sz Example Codes](./classy_sz_examples.md)



## 1. Import necessay modules
### Before computation, always import relevant modules, some of the lines below are optional:
```python
# import necessary modules
# uncomment to get plots displayed in notebook
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from classy_sz import Class as Class_sz
import math

font = {'size'   : 16, 'family':'STIXGeneral'}
axislabelfontsize='large'
matplotlib.rc('font', **font)
plt.rcParams["figure.figsize"] = [8.0,8.0]


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
```

## 2. Set cosmological parameters
There are two types of parameters in classy_sz: cosmological and astrophysical parameters. This section is about how to set cosmological parameters.

### In almost all cases, we will set our cosmological parameters in the initalization steps, for example:
```python
cosmo_params = {
'omega_b': 0.02242,
'omega_cdm':  0.11933,
'H0': 67.66, # use H0 because this is what is used by the emulators and to avoid any ambiguity when comparing with camb.
'tau_reio': 0.0561,
'ln10^{10}A_s': 3.047,
'n_s': 0.9665
}
```
Certainly more parameters can be included, see `classy_sz_parameters_and_derived_parameters.md` for additional reference.

## 3. Astrophysical parameters (Galaxies, halos, SZ)
In addition to cosmological parameters, we need to set astrophysical parameters for baryonic physics i.e. when we considering galaxies, tSZ, kSZ. For example:
- To compute tSZ power spectrum you need to specify precision parameters
- For some tSZ and kSZ calculations, you need to initialize the HOD parameters
- For some tSZ and kSZ calculations, you need to initialize the pressure profile parameters
- For kSZ calculations, you need to set kSZ parameters

### Some of the examples are included below:
### Precision parameters 
```python
precision_params = {
'x_outSZ': 4., # truncate profile beyond x_outSZ*r_s

'n_m_pressure_profile' :50, # default: 100, decrease for faster
'n_z_pressure_profile' :50, # default: 100, decrease for faster


'use_fft_for_profiles_transform' : 1, # use fft's or not.
# only used if use_fft_for_profiles_transform set to 1
'N_samp_fftw' : 512,
'x_min_gas_pressure_fftw' : 1e-4,
'x_max_gas_pressure_fftw' : 1e4,


'ndim_redshifts' :30,


'redshift_epsabs': 1.0e-40,
'redshift_epsrel': 0.0001,


'mass_epsabs': 1.0e-40,
'mass_epsrel': 0.0001
}
```

### Halo Occupation Distribution (HOD)
```python
HOD_0 = {
'sigma_log10M_HOD_ngal_0':  0.68660116,
'alpha_s_HOD_ngal_0':    1.3039425,
'M1_prime_HOD_ngal_0': 10**12.701308, # Msun/h
'M_min_HOD_ngal_0': 10**11.795964, # Msun/h
'M0_HOD_ngal_0' :0,
'x_out_truncated_nfw_profile_satellite_galaxies_ngal_0':  1.0868995,
'f_cen_HOD_ngal_0' : 1.,
}
# This is HOD_O, you can define more HODs by adding HOD_1, HOD_2, etc.

HOD_common = {
'z_min': 0.005,
'z_max': 4,
'M_min': 7.0e8,
'M_max': 3.5e15,

'dlogell': 0.2,
'ell_max': 6000.0,
'ell_min': 2.0,


'redshift_epsabs': 1.0e-40,
'redshift_epsrel': 0.0001,
'mass_epsabs': 1.0e-40,
'mass_epsrel': 0.001,
# 'ndim_masses': 150,
'ndim_redshifts': 50,

'hm_consistency': 1,

'delta for galaxies': "200c",
'delta for matter density': "200c",
'mass function': 'T08M200c',
'concentration parameter': 'B13' ,

'M0 equal M_min (HOD)':'no',
'x_out_truncated_nfw_profile': 1.0,
}
```
### Pressure profile parameters (B12, A10):
```python
# Battaglia 2012 pressure profile, can be A10 ...
b12 = {
    'pressure profile': 'B12',
    'units for tSZ spectrum': 'dimensionless',
    'x_outSZ': 4.,
    # 'truncate_gas_pressure_wrt_rvir': 0,

    'use_fft_for_profiles_transform' : 1,
    'N_samp_fftw' : 1024,
    'x_min_gas_pressure_fftw' : 1e-5,
    'x_max_gas_pressure_fftw' : 1e5,

    'pressure_profile_epsrel': 1e-3,
    'pressure_profile_epsabs': 1e-40,

    'n_z_pressure_profile': 80,
    'n_m_pressure_profile' : 80,
    'n_l_pressure_profile' : 80,

    'l_min_gas_pressure_profile' :  1.e-2,
    'l_max_gas_pressure_profile' :  5.e4,

    }
```
### kSZ parameters:
```python
# the parameters needed for the ksz calculations:
ksz_params = {
#fiducial ksz params

'k_min_for_pk_class_sz' : 0.001,
'k_max_for_pk_class_sz' : 50.0,
'k_per_decade_class_sz' : 50,
'P_k_max_h/Mpc' : 50.0,

'nfw_profile_epsabs' : 1e-33,
'nfw_profile_epsrel' : 0.001,


'ndim_masses' : 80,
'ndim_redshifts' : 30,




'n_k_density_profile' : 50,
'n_m_density_profile' : 50,
'n_z_density_profile' : 50,
'k_per_decade_for_pk' : 50,
'z_max_pk' : 4.0,

## some settings to try more points to avoid numerical noise in some cases:
## for example
# 'ndim_masses' : 100,
# 'ndim_redshifts' : 100,
# 'n_ell_density_profile' : 100,
# 'n_m_density_profile' : 100,
# 'n_z_density_profile' : 100,


# slow:
# 'n_z_psi_b1g' : 100,
# 'n_l_psi_b1g' : 400,

# 'n_z_psi_b2g' : 100,
# 'n_l_psi_b2g' : 400,

# 'n_z_psi_b2t' : 100,
# 'n_l_psi_b2t' : 400,

# 'n_z_psi_b1t' : 100,
# 'n_l_psi_b1t' : 100,

# 'n_z_psi_b1gt' : 100,
# 'n_l_psi_b1gt' : 100,


# fast:
'n_z_psi_b1g' : 50,
'n_l_psi_b1g' : 50,

'n_z_psi_b2g' : 50,
'n_l_psi_b2g' : 50,

'n_z_psi_b2t' : 50,
'n_l_psi_b2t' : 50,

'n_z_psi_b1t' : 50,
'n_l_psi_b1t' : 50,

'n_z_psi_b1gt' : 50,
'n_l_psi_b1gt' : 50,

'N_samp_fftw' : 1024, # fast: 800 ;  slow: 2000
'l_min_samp_fftw' : 1e-9,
'l_max_samp_fftw' : 1e9,

}
```

## 4. Computational parameters
 After we properly initialize following the procedures above, we move on to the computational step. 
### We need to set more **computational parameters** using ```.set``` function. There are many subtleties which need to be taken extra care in computation, as listed below:
#### Firstly, set the parameters we defined before using ```.set```, e.g.:
```python
M = Class() # Always call this
M.set(cosmo_params) # Setting the cosmological parameters
M.set(precision_params) #Setting the precision parameters
M.set(HOD_0) #Setting the HOD parameters
```
#### Secondly, we must set output parameters, depending on what you are computing, some of the examples are:
- CMB power spectrum: ```tCl,lCl,pCl```
- Matter power spectrum: ```mPk```
- Halo mass functions: ```dndlnM```
- CMB lensing cross galaxy lensing: ```gallens_lens_1h,gallens_lens_2h``` (note 1-halo and 2-halo terms)
- Galaxy-galaxy lensing: ```gamma_gal_gallens_1h,gamma_gal_gallens_2h```
- tSZ power spectrum: ```tSZ_1h,tSZ_2h```
- tSZ cross galaxies: ```galn_tsz_1h, galn_tsz_2h```
- kSZ^2 cross galaxies: ```mean_galaxy_bias,kSZ_kSZ_gal fft (1h),kSZ_kSZ_gal fft (2h),kSZ_kSZ_gal fft (3h)``` (note this is a trispectrum)
- Cluster counts: ```sz_cluster_counts```
#### Examples of setting these output parameters are:
```python
M.set({'output':'tCl,lCl,pCl'}) #CMB Power spectrum
M.set({'output':'mPk'}) # Matter power spectrum
M.set({'output': 'tSZ_1h,tSZ_2h'}) #tSZ power spectrum
M.set({'output': 'mean_galaxy_bias,kSZ_kSZ_gal fft (1h),kSZ_kSZ_gal fft (2h),kSZ_kSZ_gal fft (3h)'}) #kSZ^2 cross galaxie
"""
You can use any example commands as shown above
"""
```
### When computing angular power spectrum, you may specify the multipole range, for example:
```python
M.set({'ell_max' : 5e5, 'ell_min' : 2, 'dlogell':0.1})
```
### Depending on your calculation, you may need to set other parameters. Some of the examples are:
#### Computing the halo mass function: we need mass function, mass and redshift limit:
```python
params = {
        # mass function
        'mass_function' : 'T08M200m',
        #integration precision settings
        'ndim_redshifts' :80,
        #redshift and mass bounds
        'z_min' : 0.,
        'z_max' : 3.,
        'M_min' : 1e10,
        'M_max' : 1e15,

}
M.set(params)
```
#### Computing the galaxy x galaxy power spectrum: need a lot of inputs:
```python
M.set({
'ndim_masses': 80,
'ndim_redshifts':80,

'mass_function':'T08M200c',


'M_min':1e11, # Msun/h
'M_max':1e15, # Msun/h
'mass_epsrel': 1e-3,
'mass_epsabs': 1e-40,
'z_min':1e-5,
'z_max': 2.,
'redshift_epsrel': 0.5e-3,
'redshift_epsabs': 1e-40,

'delta_for_galaxies':'200c',
'delta_for_matter_density':'200c',

'concentration_parameter':'fixed',

'M0_HOD': 0.,
'M_min_HOD':10.**11.57, #Msun/h
'M1_prime_HOD':10.**12.75, #Msun/h

'sigma_log10M_HOD':0.17,
'alpha_s_HOD':0.99,

'x_out_truncated_nfw_profile_satellite_galaxies':1., # so corresponds to 1xr200c

'csat_over_cdm' : 1.,

'f_cen_HOD': 1.,
'Delta_z_lens':0.00,
'Delta_z_source':0.00,

'galaxy_sample' : 'custom',
'full_path_to_dndz_gal' : path_to_class_sz+'/class_sz/class-sz/class_sz_auxiliary_files/includes/nz_lens_bin1.txt', # lens galaxies
'full_path_to_source_dndz_gal' : path_to_class_sz+'/class_sz/class-sz/class_sz_auxiliary_files/includes/nz_source_normalized_bin4.txt', # source galaxies

'N_samp_fftw':1024, #precision parameter for the bessel transform to theta space
'l_min_samp_fftw' : 1e-8,
'l_max_samp_fftw' : 1e8,

'hm_consistency' : 1,
# 'ndim_redshifts': 200,
# 'ndim_masses': 500,

'use_pknl_in_2hterms': 0,


# 'P_k_max_h/Mpc':5e1,
# 'non_linear':'halofit',

'do_real_space_with_mcfit': 1,
})
M.compute_class_szfast()
```

#### Computing tSZ power spectrum: we need mass function, mass and redshift limit, pressure profile:
```python
M.set({
'z_min' : 0.005,
'z_max' : 3.0,
'M_min' : 1.0e10,
'M_max' : 3.5e15,
'mass_function' : 'T08M500c',
'pressure_profile':'GNFW', # can be Battaglia, Arnaud, etc

"P0GNFW": 8.130,
"c500": 1.156,
"gammaGNFW": 0.3292,
"alphaGNFW": 1.0620,
"betaGNFW":5.4807,
})
```
#### Computing tSZ cross galaxies: we need the galaxy samples and their file path:
```python
M.set({
'galaxy_samples_list_num' : 3, # the number of galaxy samples
'galaxy_samples_list' : '0,1,2', # the id string of each sample, can be any integer
'full_path_and_prefix_to_dndz_ngal':path_to_class_sz + '/class_sz/class-sz/class_sz_auxiliary_files/includes/normalised_dndz_cosmos_'
})
```

#### Computing kSZ^2 cross galaxies: we need the gas profile:
```python
M.set({
'output':'mean_galaxy_bias,kSZ_kSZ_gal fft (1h),kSZ_kSZ_gal fft (2h),kSZ_kSZ_gal fft (3h)',
'projected_field_filter_file' : path_to_class_sz + 'class_sz_auxiliary_files/includes/s4_fl_A_170422.txt',

'gas_profile' : 'B16', # set Battaglia 2016 density profile
'gas_profile_mode' : 'agn',
'normalize_gas_density_profile' : 0,
'use_xout_in_density_profile_from_enclosed_mass' : 1,


## with current settings, the calculation seems more stable without use_fft_for_profiles_transform
## i.e., 'use_fft_for_profiles_transform' : 0
'use_fft_for_profiles_transform' : 0,

'use_bg_at_z_in_ksz2g_eff' : 1,
'non_linear' : 'halofit',
      })
```
and many more examples.
## 5. After you set all parameters, run:
```python
M.compute_class_szfast()
```

### This completes the initialization of classy_sz.

## 6. Computing the output result

### To compute the result, you need to call specific functions depending on what you need to compute. Examples of those functions are:
#### CMB power spectrum: ```M.lensed_cl()```:
```python
lensed_cls = M.lensed_cl()
l_fast = lensed_cls['ell']
cl_tt_fast = lensed_cls['tt']
cl_ee_fast = lensed_cls['ee']
cl_te_fast = lensed_cls['te']
cl_pp_fast = lensed_cls['pp']
```
#### Linear matter density power spectrum: ```M.pk_lin(ks,z)```

#### CMB lensing cross galaxy lensing: ```M.cl_kg_k()```
```python
ell =  np.asarray(M.cl_kg_k()['ell'])
cl1h = M.cl_kg_k()['1h']
cl2h = M.cl_kg_k()['2h']
```
#### tSZ power spectrum: ```M.cl_sz()```
```python
l = np.asarray(classy_sz.cl_sz()['ell'])
cl_yy_1h = np.asarray(classy_sz.cl_sz()['1h'])
cl_yy_2h = np.asarray(classy_sz.cl_sz()['2h'])
```
#### tSZ cross galaxies: ```M.cl_galn_tsz()```
#### kSZ^2 cross galaxies: ```M.cl_kSZ_kSZ_g()```

### Note in the example codes above how the different elements (ell, cl1h, cl2h, etc.) are indices in the array.


## Appendix: Some useful functions
### Functions for Computing the Auto and Cross Power Spectrum

#### Auto Power Spectrum (CMB, Lensing, tSZ, kSZ)
| Calculation                              | Output Key(s)        | Function Call              | Description                                                                                 |
|------------------------------------------|----------------------|----------------------------|---------------------------------------------------------------------------------------------|
| CMB temperature and polarization power spectrum | ```tCl, lCl, pCl``` | ```M.lensed_cl()```        | Standard CMB power spectra (TT, EE, BB, TE, lensing, etc.).                                 |
| CMB lensing convergence spectrum         | ```?```              | ```M.cl_c1_lens()```       | Computes CMB lensing convergence auto power spectrum (1-halo & 2-halo).                    |
| Matter power spectrum                    | ```mPk```            | ```M.pk_lin(k,z)``` (linear) or ```M.pk(k,z)``` (non-linear) | Computes linear or non-linear matter power spectrum ```P(k)```.                           |
| tSZ power spectrum                       | ```tSZ_1h, tSZ_2h``` | ```M.cl_sz()```            | Thermal Sunyaev-Zel’dovich power spectrum (1-halo & 2-halo).                                |
| kSZ power spectrum                       | ```?```              | ```M.cl_ksz()```           | Computes the 1-halo and 2-halo terms of the kSZ × kSZ power spectrum.                       |
| Lensing magnification power spectrum     | ```?```              | ```M.cl_mm()```            | Computes the 1-halo, 2-halo, and higher-order terms of the lensing magnification power spectrum. |
| Lensing convergence power spectrum       | ```?```              | ```M.cl_kk()```            | Computes the 1-halo, 2-halo, and higher-order terms of the lensing convergence power spectrum. |
| Electron x Electron power spectrum       | ```?```              | ```M.cl_ee()```            | Computes the 1-halo and 2-halo terms of the electron x electron power spectrum.             |

#### Cross Power Spectrum (Correlations)
| Calculation                              | Output Key(s)                                  | Function Call              | Description                                                                                 |
|------------------------------------------|----------------------------------------------|----------------------------|---------------------------------------------------------------------------------------------|
| CMB lensing x Galaxy lensing             | ```gallens_lens_1h, gallens_lens_2h```       | ```M.cl_kg_k()```          | 1-halo & 2-halo cross-correlation of CMB lensing with galaxy lensing.                       |
| Galaxy-galaxy lensing                    | ```gamma_gal_gallens_1h, gamma_gal_gallens_2h``` | ```M.gamma_ggamma()```     | 1-halo & 2-halo galaxy-galaxy lensing signals (tangential shear).                           |
| tSZ x Galaxies                           | ```galn_tsz_1h, galn_tsz_2h```               | ```M.cl_galn_tsz()```      | Cross-correlation of tSZ with galaxies (1-halo & 2-halo).                                   |
| kSZ^2 x Galaxies                         | ```mean_galaxy_bias, kSZ_kSZ_gal fft (1h,2h,3h)``` | ```M.cl_kSZ_kSZ_g()```     | kSZ^2–galaxy cross (trispectrum). Outputs 1h, 2h, 3h contributions.                         |
| Galaxy lensing x Galaxy lensing          | ```?```                                      | ```M.cl_kg_kg()```         | Computes the 1-halo and 2-halo terms of galaxy lensing × galaxy lensing power spectrum.     |
| Galaxy lensing x CMB lensing             | ```?```                                      | ```M.cl_kg_k()```          | Computes the 1-halo and 2-halo terms of galaxy lensing × CMB lensing power spectrum.        |
| tSZ x Galaxy lensing                     | ```?```                                      | ```M.cl_ykg()```           | Computes the 1-halo and 2-halo terms of tSZ × galaxy lensing power spectrum.                |
| IA x Galaxy                              | ```?```                                      | ```M.cl_IA_g()```          | Computes the 2-halo term of intrinsic alignment × galaxy power spectrum.                   |
| Galaxy lensing x Lensing magnification   | ```?```                                      | ```M.cl_kg_m()```          | Computes the 1-halo and 2-halo terms of galaxy lensing × lensing magnification power spectrum. |
| tSZ x Lensing magnification              | ```?```                                      | ```M.cl_ym()```            | Computes the 1-halo and 2-halo terms of tSZ × lensing magnification power spectrum.         |
| kSZ^2 x kSZ^2 x Galaxy                   | ```?```                | ```M.cl_kSZ_kSZ_g()```     | Computes the 1-halo, 2-halo, and 3-halo terms of kSZ^2 × kSZ^2 × galaxy power spectrum.     |
| Galaxy x Galaxy x CMB lensing            | ```?```                             | ```M.cl_gal_gal_kcmb()```  | Computes the 1-halo, 2-halo, and 3-halo terms of galaxy × galaxy × CMB lensing power spectrum. |
| Te x tSZ x tSZ                           | ```?```                                   | ```M.cl_te_y_y()```        | Computes the Te × tSZ × tSZ power spectrum.                                                |
| kappa x Galaxy                           | ```?```                             | ```M.cl_kg()```            | Computes the 1-halo and 2-halo terms of kappa (lensing) × galaxy power spectrum.    |

Note: for some of the output keys above which are missing, labelled as ```?```, you may deduce them via regular expression, the relevant code in the source codes are:
```python
# Define the regular expression pattern
pattern1 = re.compile(r'^[^,]+_(1h|2h|hf|3h|covmat|lensing_term|hsv)$')
pattern2 = re.compile(r'^m\d+[a-zA-Z]*_to_m\d+[a-zA-Z]*$')
pattern3 = re.compile(r'.*\((1h|2h|3h)\)$')
```
#### Other Functions

| Calculation             | Output Key(s)         | Function Call                                   | Description                                                                 |
|-------------------------|-----------------------|------------------------------------------------|-----------------------------------------------------------------------------|
| Halo Mass Function       | ```dndlnM```          | ```M.get_nu_at_z_and_m(), M.get_first_order_bias_at_z_and_nu(), M.get_second_order_bias_at_z_and_nu()``` | Computes the halo mass function and related quantities like bias parameters. |
| Cluster Counting         | ```sz_cluster_counts``` | ```class_sz.dndzdy_theoretical()```            | Computes the theoretical cluster counts in redshift and mass bins.         |

