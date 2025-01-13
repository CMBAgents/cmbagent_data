
# Examples for some computations:
## Example code for matter power spectrum:
- Interpolator: use ```classy_sz.pk_lin(ks,z)```
- On the k-grid method: use ```classy_sz.get_pkl_at_z(z,params_values_dict = cosmo_params)```
```python
classy_sz = Class_sz()
classy_sz.set(cosmo_params)
classy_sz.set({
'output':'mPk',
'ndim_redshifts': 25
})
classy_sz.compute_class_szfast()

# Method 1: interpolator
kmin = 1e-3
kmax = 1e1
nks = 500
ks = np.geomspace(kmin,kmax,nks)
pks = classy_sz.pk_lin(ks,z)

# Method 2: On the k-grid
z = 0.3
pks,ks = classy_sz.get_pkl_at_z(z,params_values_dict = cosmo_params)
```

## Example code for tSZ cross galaxy lensing:
```python
#Initialize the HOD
# best-fit from Kusiak et al. https://arxiv.org/pdf/2203.12583.pdf

HOD_0 = {
'sigma_log10M_HOD_ngal_0': 0.68660116,
'alpha_s_HOD_ngal_0':    1.3039425,
'M1_prime_HOD_ngal_0': 10**12.701308, # Msun/h
'M_min_HOD_ngal_0': 10**11.795964, # Msun/h
'M0_HOD_ngal_0' :0,
'x_out_truncated_nfw_profile_satellite_galaxies_ngal_0':  1.0868995,
'f_cen_HOD_ngal_0' : 1.,
}

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

# Initialize the pressure profile
# Battaglia 2012 pressure profile
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
    
# Provide galaxy redshift distribution and save as correct format

import os
path_to_class_sz = os.environ['PATH_TO_CLASS_SZ_DATA']
import numpy as np
zuw, dnblue = np.loadtxt(path_to_class_sz + '/class_sz/class-sz/class_sz_auxiliary_files/includes/normalised_dndz_cosmos.txt',unpack=True)
np.savetxt(path_to_class_sz + '/class_sz/class-sz/class_sz_auxiliary_files/includes/normalised_dndz_cosmos_0.txt',np.c_[zuw,dnblue])

# Computation
M = Class()
M.set(common_settings)
M.set(HOD_common)
M.set(HOD_0) #ngal_0
M.set(HOD_1) #ngal_1
M.set(HOD_2) #ngal_2
M.set(b12) # Battaglia pressure profile

M.set({
'output' : 'galn_tsz_1h, galn_tsz_2h', # lensmagn_tsz also available
'galaxy_samples_list_num' : 3, # the number of galaxy samples
'galaxy_samples_list' : '0,1,2', # the id string of each sample, can be any integer
'full_path_and_prefix_to_dndz_ngal':path_to_class_sz + '/class_sz/class-sz/class_sz_auxiliary_files/includes/normalised_dndz_cosmos_'
})


M.compute_class_szfast()
# Collect and plot
cl_galn_tsz =  M.cl_galn_tsz()
dict_names = {
'0':r'$\mathrm{Blue}$','1':r'$\mathrm{Green}$','2':r'$\mathrm{Red}$'
}

plot_dim = len(cl_galn_tsz.keys())
fig, axes = plt.subplots(figsize=(7, 15),
                         sharex=True,
                         sharey=True,
                         nrows=plot_dim
                         )
plt.subplots_adjust(#left=0.1,
#                     bottom=0.1,
#                     right=0.9,
#                     top=0.9,
#                     wspace=0.4,
                    hspace=0.4)
# x = np.linspace(0, 10, 100)
ik = 0
for i in range(plot_dim):
    kk = list(cl_galn_tsz.keys())[ik]
    strp = list(cl_galn_tsz.keys())[ik]
    kt = dict_names[strp.split('x')[0]]
    ik+=1
    # axes[i, j].plot(x, np.sin((i+j) *x))
    axes[i].tick_params(axis='both', which='major', labelsize=15)
    axes[i].set_title(kt,size=20)
    axes[i].set_xlabel(r"$\ell$",fontsize=20)
    axes[i].set_ylabel(r"$C_l^{g \times y}$",fontsize=22)

    axes[i].grid()

    cl = cl_galn_tsz[kk]
    fac = np.asarray(cl['ell'])*(np.asarray(cl['ell'])+1.)/2./np.pi

    axes[i].loglog(cl['ell'],np.asarray(cl['1h'])/fac,":",color ='black',label=r'1h')
    axes[i].loglog(cl['ell'],np.asarray(cl['2h'])/fac,"--",color ='black',label=r'2h')
    axes[i].loglog(cl['ell'],(np.asarray(cl['1h'])+np.asarray(cl['2h']))/fac,color ='red',label=r'total')
    axes[i].legend(loc=1,fontsize=8)

fig.suptitle('CLASS_SZ Galaxy-tSZ (HOD, B12)',fontsize=20)
fig.tight_layout()
# plt.savefig('../../notebooks/class_sz_tutorial_notebooks/figures/class_sz_unwise_hod_gkcmb.pdf')

```

## Example code for CMB lensing x galaxy lensing:
### In this computation we need to supply the distribution and redshift of source galaxies ```z, nz_sources```
### ```M.cl_kg_k``` means to compute $C_l^{\kappa\gamma}$, the harmonic space angular power spectra of CMB lensing cross cosmic shear.
```python
# Provide galaxy kernels
# z , nz_lenses = np.loadtxt(path_to_class_sz+'/class_sz/class-sz/class_sz_auxiliary_files/includes/nz_lens_bin1.txt',unpack=True)
z , nz_sources = np.loadtxt(path_to_class_sz+'/class_sz/class-sz/class_sz_auxiliary_files/includes/nz_source_normalized_bin4.txt',unpack=True)

# Computation
M = Class()
M.set(cosmology)
M.set({
'output':'gallens_lens_1h,gallens_lens_2h',
'ell_max' : 5e3,
'ell_min' : 2,
'dlogell':0.1,
'ndim_redshifts':80,

'mass_function':'T08M200c',


'M_min':1e11, # Msun/h
'M_max':1e15, # Msun/h
# 'mass_epsrel': 1e-3,
# 'mass_epsabs': 1e-40,

'z_min':1e-5,
'z_max': 2.,
# 'redshift_epsrel': 0.5e-3,
# 'redshift_epsabs': 1e-40,

'delta_for_galaxies':'200c',
'delta_for_matter_density':'200c',

'concentration_parameter':'fixed',


'Delta_z_source':0.00,

'galaxy_sample' : 'custom',
'full_path_to_source_dndz_gal' : path_to_class_sz+'/class_sz/class-sz/class_sz_auxiliary_files/includes/nz_source_normalized_bin4.txt', # source galaxies

'N_samp_fftw':1024, #precision parameter for the bessel transform to theta space
'l_min_samp_fftw' : 1e-8,
'l_max_samp_fftw' : 1e8,

'hm_consistency' : 1,

})
M.compute_class_szfast()
ell =  np.asarray(M.cl_kg_k()['ell'])
cl1h = M.cl_kg_k()['1h']
cl2h = M.cl_kg_k()['2h']
```
## Example code for galaxy-galaxy lensing:
### We need to supply both the source and lens galaxy redshift distributions ```z , nz_sources``` and ```z , nz_lenses```.
### ```M.cl_ggamma()``` means to compute $C_l^{g\gamma}$, the harmonic space angular power spectrum between galaxy overdensity and cosmic shear.
```python
z , nz_lenses = np.loadtxt(path_to_class_sz+'/class_sz/class-sz/class_sz_auxiliary_files/includes/nz_lens_bin1.txt',unpack=True)
z , nz_sources = np.loadtxt(path_to_class_sz+'/class_sz/class-sz/class_sz_auxiliary_files/includes/nz_source_normalized_bin4.txt',unpack=True)
plt.plot(z,nz_sources,label='sources (normalized)',ls='--')
M = Class()
# M.set(cosmology)
M.set(fast_cosmology)
M.set({
'output':'gamma_gal_gallens_1h,gamma_gal_gallens_2h',
'ell_max' : 5e5,
'ell_min' : 2,
'dlogell':0.1,

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
cl_g_gamma_ell = np.asarray(M.cl_ggamma()['ell'])
cl_g_gamma_1h = np.asarray(M.cl_ggamma()['1h'])
cl_g_gamma_2h = np.asarray(M.cl_ggamma()['2h'])
```




## Example code for kSZ^2 cross galaxies trispectrum
```python

common_settings = {

                   'H0':67.556,
                   'omega_b':0.022032,
                   'omega_cdm':0.12038,
                   'ln10^{10}A_s': 3.047,
                   'n_s': 0.9665,
                   'tau_reio':0.0925,
                   'cosmo_model': 0, # This parameter is important if you want to compute with emulators. Set to 0 for lcdm with \Sigma mnu=0.06 eV.

}

# best-fit from Kusiak et al. https://arxiv.org/pdf/2203.12583.pdf

HOD_blue = {
'sigma_log10M_HOD': 0.68660116,
'alpha_s_HOD':    1.3039425,
'M1_prime_HOD': 10**12.701308, # Msun/h
'M_min_HOD': 10**11.795964, # Msun/h
'M0_HOD' :0,
'x_out_truncated_nfw_profile_satellite_galaxies':  1.0868995,
'f_cen_HOD' : 1.,
'full_path_to_dndz_gal': path_to_class_sz + 'class_sz_auxiliary_files/includes/normalised_dndz_cosmos_0.txt',
}


unWISE_common = {
'galaxy_sample': 'custom',
'M0_equal_M_min_HOD':'no',
'x_out_truncated_nfw_profile': 1.0,


'z_min': 0.005,
'z_max': 3.,
'M_min': 1e10,
'M_max': 3.5e15,

'nfw_profile_epsabs' : 1e-33,
'nfw_profile_epsrel' : 0.001,


'x_min_gas_density_fftw' : 1e-5,
'x_max_gas_density_fftw' : 1e4,


'redshift_epsabs': 1.0e-40,
'redshift_epsrel': 0.001,
'mass_epsabs': 1.0e-40,
'mass_epsrel': 0.001,



'hm_consistency': 1,


'delta_for_galaxies': "200c",
'delta_for_matter_density': "200c",
'mass_function': 'T08M200c',
'concentration parameter': 'B13' ,


}

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
%%time
M = Class()
M.set(common_settings)
M.set(HOD_blue)
M.set(unWISE_common)
M.set(ksz_params)
M.set({
'output':'mean_galaxy_bias,kSZ_kSZ_gal fft (1h),kSZ_kSZ_gal fft (2h),kSZ_kSZ_gal fft (3h)',


'projected_field_filter_file' : path_to_class_sz + 'class_sz_auxiliary_files/includes/s4_fl_A_170422.txt',

'dlogell' : 0.1,
'ell_max' : 10000.0,
'ell_min' : 10.0,

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

M.compute_class_szfast()
cl_kSZ_kSZ_g = M.cl_kSZ_kSZ_g()

label_size = 17
title_size = 18
legend_size = 13
handle_length = 1.5
fig, (ax1) = plt.subplots(1,1,figsize=(7,4))
ax = ax1
ax.tick_params(axis = 'x',which='both',length=5,direction='in', pad=10)
ax.tick_params(axis = 'y',which='both',length=5,direction='in', pad=5)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
plt.setp(ax.get_yticklabels(), rotation='horizontal', fontsize=label_size)
plt.setp(ax.get_xticklabels(), fontsize=label_size)
ax.grid( visible=True, which="both", alpha=0.2, linestyle='--')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1e-4,1.5e-1)
ax.set_xlim(50,9000)

fac = (2.726e6)**2*np.asarray(cl_kSZ_kSZ_g['ell'])*(np.asarray(cl_kSZ_kSZ_g['ell'])+1.)/2./np.pi

ax.plot(cl_kSZ_kSZ_g['ell'],fac*np.asarray(cl_kSZ_kSZ_g['1h']),label = r'$1\mathrm{h}$',c='k',ls=':',lw=0.7)
ax.plot(cl_kSZ_kSZ_g['ell'],fac*np.asarray(cl_kSZ_kSZ_g['2h']),label = r'$2\mathrm{h}$',c='b',ls='-.',lw=0.7)
ax.plot(cl_kSZ_kSZ_g['ell'],fac*np.asarray(cl_kSZ_kSZ_g['3h']),label = r'$3\mathrm{h}$',c='orange',ls='--',lw=0.7)
ax.plot(cl_kSZ_kSZ_g['ell'],fac*(np.asarray(cl_kSZ_kSZ_g['1h'])+np.asarray(cl_kSZ_kSZ_g['2h'])+np.asarray(cl_kSZ_kSZ_g['3h'])),
        label = r'$\mathrm{Total}$',c='k',ls='-',lw=1.)

ax.legend(loc=2,ncol = 1,frameon=False,fontsize=14)
ax.set_xlabel(r"$\ell$",size=title_size)
ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{\mathrm{kSZ^2g}}/2\pi\quad [\mathrm{\mu K^2}]$",size=title_size)
fig.tight_layout()
```
