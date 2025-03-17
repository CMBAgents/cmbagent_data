# How to compute the comoving distance with classy_sz

We always assume a spatially flat universe. So the comoving distance $\chi$ is related to the angular diameter distance $D_A$ as:


$$
\chi = D_A * (1+z)
$$

where $D_A$ is the angular diameter distance computed with `classy_sz.get_angular_distance_at_z` in units of $[Mpc]$, see [classy_sz_angular_diameter_distance.md](classy_sz_angular_diameter_distance.md).

To get the comoving distance to the last scattering surface, you should use the `get_derived_parameters` method, see [classy_sz_input_and_derived_parameters.md](classy_sz_input_and_derived_parameters.md).








