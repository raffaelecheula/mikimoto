# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import timeit
import numpy as np
import matplotlib.pyplot as plt
from mikimoto import units
from mikimoto.microkinetics import (
    Species, Solution, Interface, Reaction, Microkinetics,
)
from mikimoto.thermodynamics import ThermoConstantCp, ThermoFixedG0
from mikimoto.reactors import (
    BatchReactor, CSTReactor, PFReactor, PFReactorSeriesCSTR,
)

# -----------------------------------------------------------------------------
# PARAMETERS
# -----------------------------------------------------------------------------

# Set temperature and pressure of the simulation.
temperature = 300 # [K]
pressure = 1 * units.atm # [Pa]

# Set the integration parameters.
n_cstr = 100

# Set the catalyst parameters.
cat_mass = 0.10 * units.gram # [kg]
cat_mol_weight = 118.41 * units.gram/units.mole # [kg/kmol]
cat_dispersion = 0.10 # [kmol/kmol]

# Set the volumetric flow rate from gas hour space velocity.
vol_flow_rate = 24000 * units.milli*units.litre/units.hour # [m^3/s/g]

# Set the cross section and the total volume of the reactor.
reactor_length = 10.00 * units.centimeter # [m]
diameter = 1.00 * units.centimeter # [m]

# Calculate reactor volume and gas velocity.
cross_section = np.pi*(diameter**2)/4. # [m^2]
reactor_volume = cross_section*reactor_length # [m^3]
gas_velocity = vol_flow_rate/cross_section # [m/s]

# Calculate the catalyst active area per unit of reactor volume.
cat_moles = cat_mass/cat_mol_weight # [kmol]
cat_sites_conc = cat_moles*cat_dispersion/reactor_volume # [kmol/m^3]

# -----------------------------------------------------------------------------
# MICROKINETIC MODEL
# -----------------------------------------------------------------------------

units_energy = units.eV/units.molecule

Gibbs_gas_species = {
    'CO2': +0.0 * units_energy,
    'H2': +0.0 * units_energy,
    'CO': +0.0 * units_energy,
    'H2O': +0.0 * units_energy,
    'N2': +0.0 * units_energy,
}

Gibbs_gas_reactions = {
}

Gibbs_surf_species = {
    '(X)': +0.0 * units_energy,
    'CO2(X)': +0.3 * units_energy,
    'CO(X)': +0.1 * units_energy,
    'O(X)': +0.1 * units_energy,
    'H(X)': +0.1 * units_energy,
    #'COOH(X)': +0.1 * units_energy,
    'OH(X)': +0.1 * units_energy,
    'H2O(X)': +0.1 * units_energy,
}

Gibbs_surf_reactions = {
    'CO2 + (X) <=> CO2(X)': +0.2 * units_energy,
    'H2 + (X) + (X) <=> H(X) + H(X)': +0.2 * units_energy,
    'CO2(X) + (X) <=> CO(X) + O(X)': +1.0 * units_energy,
    #'CO2(X) + H(X) <=> COOH(X) + (X)': +1.0 * units_energy,
    #'COOH(X) + (X) <=> CO(X) + OH(X)': +0.7 * units_energy,
    'O(X) + H(X) <=> OH(X) + (X)': +0.2 * units_energy,
    'OH(X) + H(X) <=> H2O(X) + (X)': +0.2 * units_energy,
    'CO + (X) <=> CO(X)': +0.2 * units_energy,
    'H2O + (X) <=> H2O(X)': +0.2 * units_energy,
}

x_in_dict = {
    'CO2': 0.30,
    'H2': 0.30,
    'CO': 0.10,
    'H2O': 0.10,
    'N2': 0.20,
}

θ_in_dict = {
    '(X)': 1.00,
    'CO2(X)': 0.00,
    'CO(X)': 0.00,
    'O(X)': 0.00,
    'H(X)': 0.00,
    'COOH(X)': 0.00,
    'OH(X)': 0.00,
    'H2O(X)': 0.0,
}

# Example 2
"""
Gibbs_gas_species = {
    'COO': +0.0 * units_energy,
    'OOC': -0.3 * units_energy,
}

Gibbs_gas_reactions = {
}

Gibbs_surf_species = {
    '(X)': +0.0 * units_energy,
    'COO(X)': +0.6 * units_energy,
    'OCO(X)': +0.6 * units_energy,
    'OOC(X)': +0.1 * units_energy,
}

Gibbs_surf_reactions = {
    'COO + (X) <=> COO(X)': +0.20 * units_energy,
    'COO(X) <=> OCO(X)': +0.65 * units_energy,
    'OCO(X) <=> OOC(X)': +0.60 * units_energy,
    'OOC(X) <=> OOC + (X)': +0.20 * units_energy,
}

x_in_dict = {
    'COO': 0.80,
    'OOC': 0.20,
}

θ_in_dict = {
    '(X)': 1.00,
    'COO(X)': 0.00,
    'OCO(X)': 0.00,
    'OOC(X)': 0.00,
}
"""

gas_species = [
    Species(name, thermo = ThermoFixedG0(Gibbs_ref = Gibbs_gas_species[name]))
    for name in Gibbs_gas_species
]
gas_reactions = [
    Reaction(name, thermo = ThermoFixedG0(Gibbs_ref = Gibbs_gas_reactions[name]))
    for name in Gibbs_gas_reactions
]

gas = Solution(
    temperature = temperature,
    pressure = pressure,
    species = gas_species,
    reactions = gas_reactions,
)

surf_species = [
    Species(name, thermo = ThermoFixedG0(Gibbs_ref = Gibbs_surf_species[name]))
    for name in Gibbs_surf_species
]
surf_reactions = [
    Reaction(name, thermo = ThermoFixedG0(Gibbs_ref = Gibbs_surf_reactions[name]))
    for name in Gibbs_surf_reactions
]

surf = Interface(
    temperature = temperature,
    pressure = pressure,
    sites_conc = cat_sites_conc,
    species = surf_species,
    reactions = surf_reactions,
)

microkin = Microkinetics(
    gas_phases = [gas],
    surf_phases = [surf],
    temperature = temperature,
    pressure = pressure,
)

x_in = [x_in_dict[spec.name] for spec in gas_species]
θ_in = [θ_in_dict[spec.name] for spec in surf_species]

reactor_type = None

start = timeit.default_timer()

if reactor_type == 'BatchReactor':

    batch = BatchReactor(
        microkin = microkin,
        method = 'BDF',
        reactor_volume = reactor_volume,
        t_bound = 1.,
        norm_dconc_thr = 0.,
    )
    batch.integrate_volume(x_in = x_in, θ_in = θ_in)

elif reactor_type == 'CSTReactor':

    cstr = CSTReactor(
        microkin = microkin,
        method = 'BDF',
        reactor_volume = reactor_volume,
        vol_flow_rate = vol_flow_rate,
        t_bound = 1.,
        norm_dconc_thr = 0.,
    )
    cstr.integrate_volume(x_in = x_in, θ_in = θ_in)

elif reactor_type == 'PFReactor':

    pfr = PFReactor(
        microkin = microkin,
        reactor_volume = reactor_volume,
        vol_flow_rate = vol_flow_rate,
    )
    x_out = pfr.integrate_volume(x_in = x_in, θ_in = θ_in)

elif reactor_type == 'PFReactorSeriesCSTR':

    pfr = PFReactorSeriesCSTR(
        microkin = microkin,
        n_cstr = n_cstr,
        reactor_volume = reactor_volume,
        reactor_length = reactor_length,
        vol_flow_rate = vol_flow_rate,
    )
    pfr.integrate_volume(x_in = x_in, θ_in = θ_in)

print(f'Execution time: {timeit.default_timer()-start:9.4f} [s]')

# -----------------------------------------------------------------------------
# POSTPROCESSING
# -----------------------------------------------------------------------------

calculate_DRC = True

contact_time_DRC = 1e-9
delta_e_DRC = 1e-3 * units_energy # [eV]
###species_DRC = 'OOC'
species_DRC = 'CO'

#contact_time_DRC = 1e9

cstr = CSTReactor(
    microkin = microkin,
    method = 'BDF',
    contact_time = contact_time_DRC,
    rtol = 1e-13,
    atol = 1e-15,
)

if calculate_DRC is True:
    
    print("\n ★ Degree of rate control reactions:")
    
    # TODO: Check if it is more correct to use molar fluxex instead 
    # of molar fractions.
    index_spec_DRC = microkin.species_indices[species_DRC]
    DRC_dict = {}
    x_out, θ_out = cstr.integrate_volume(x_in = x_in, θ_in = θ_in)
    dx_orig = x_out[index_spec_DRC]-x_in[index_spec_DRC]
    for react in microkin.reactions:
        react.thermo.modify_energy(delta_energy = delta_e_DRC)
        x_out, θ_out = cstr.integrate_volume(x_in = x_in, θ_in = θ_in)
        react.thermo.modify_energy(delta_energy = -delta_e_DRC)
        dx_mod = x_out[index_spec_DRC]-x_in[index_spec_DRC]
        DRC_species = (
            -(dx_mod-dx_orig)/dx_orig / 
            (delta_e_DRC/(units.Rgas)/temperature)
        )
        DRC_dict[react.name] = DRC_species

    for react in DRC_dict:
        print(f"{react:50s} = {DRC_dict[react]:+5.3f}")

    print(f"Total: {sum([DRC_dict[react] for react in DRC_dict]):+5.3f}")

calculate_DRC_ads = False

if calculate_DRC_ads is True:
    
    print("\n ★ Degree of rate control species:")
    
    index_spec_DRC = microkin.species_indices[species_DRC]
    DRC_dict = {}
    x_out, θ_out = cstr.integrate_volume(x_in = x_in, θ_in = θ_in)
    dx_orig = x_out[index_spec_DRC]-x_in[index_spec_DRC]
    for react in microkin.species:
        react.thermo.modify_energy(delta_energy = delta_e_DRC)
        x_out, θ_out = cstr.integrate_volume(x_in = x_in, θ_in = θ_in)
        react.thermo.modify_energy(delta_energy = -delta_e_DRC)
        dx_mod = x_out[index_spec_DRC]-x_in[index_spec_DRC]
        DRC_species = (
            -(dx_mod-dx_orig)/dx_orig / 
            (delta_e_DRC/(units.Rgas)/temperature)
        )
        DRC_dict[react.name] = DRC_species

    for spec in DRC_dict:
        print(f"{spec:50s} = {DRC_dict[spec]:+5.3f}")

    print(f"Total: {sum([DRC_dict[spec] for spec in DRC_dict]):+5.3f}")

x_out, θ_out = cstr.integrate_volume(x_in = x_in, θ_in = θ_in)

print("\n ★ Gas molar fractions:")
Gibbs_gas = {}
for ii, spec in enumerate(gas_species):
    print(f"x_{spec.name:10s} = {x_out[ii]:+7.4e}")
    spec.thermo.conc = x_out[ii]*pressure/(units.Rgas*temperature)
    Gibbs_gas[spec.name] = spec.thermo.Gibbs/units_energy

print("\n ★ Adsorbates coverages:")
Gibbs_ads = {}
for ii, spec in enumerate(surf_species):
    print(f"θ_{spec.name:10s} = {θ_out[ii]:+7.4e}")
    spec.thermo.conc = θ_out[ii]*cat_sites_conc
    Gibbs_ads[spec.name] = spec.thermo.Gibbs/units_energy

Gibbs_O_1 = Gibbs_gas['CO2']-Gibbs_gas['CO']
Gibbs_O_2 = Gibbs_gas['H2O']-Gibbs_gas['H2']
alpha = 0.
Gibbs_O = (Gibbs_O_1+alpha*Gibbs_O_2)/(1+alpha)
Gibbs_ref_for = {
    'X': 0.,
    'O': Gibbs_O_2,
    'H': Gibbs_gas['H2']/2.,
    'C': Gibbs_gas['CO2']-2*Gibbs_O,
}
Gibbs_ref_rev = {
    'X': 0.,
    'O': Gibbs_O_2,
    'H': (Gibbs_gas['H2O']-Gibbs_O)/2.,
    'C': Gibbs_gas['CO']-Gibbs_O,
}

"""
Gibbs_ref_for = {
    'X': 0.,
    'O': 0.,
    'C': Gibbs_gas['COO'],
}
Gibbs_ref_rev = {
    'X': 0.,
    'O': 0.,
    'C': Gibbs_gas['COO'],
}
"""

conc_tot = pressure/units.Rgas/temperature

print("\n ★ Reaction rates:")
rates_for = {}
rates_rev = {}
rates_tot = {}
for react in microkin.reactions:
    ΔGibbs_for = react.thermo.Gibbs_std/units_energy
    ΔGibbs_rev = react.thermo.Gibbs_std/units_energy
    for elem in react.composition:
        ΔGibbs_for -= react.composition[elem] * Gibbs_ref_for[elem]
        ΔGibbs_rev -= react.composition[elem] * Gibbs_ref_rev[elem]
    rates_for[react.name] = +units.kB*temperature/units.hP * conc_tot * θ_out[0] * (
        np.exp(-ΔGibbs_for*units_energy/(units.Rgas*temperature))
    )
    rates_rev[react.name] = -units.kB*temperature/units.hP * conc_tot * θ_out[0] * (
        np.exp(-ΔGibbs_rev*units_energy/(units.Rgas*temperature))
    )
    rates_tot[react.name] = rates_for[react.name]+rates_rev[react.name]
    print(f"{react.name:50s} for {rates_for[react.name]:+7.4e} [kmol/m^3/s]")
    print(f"{react.name:50s} rev {rates_rev[react.name]:+7.4e} [kmol/m^3/s]")
    print(f"{react.name:50s} tot {rates_tot[react.name]:+7.4e} [kmol/m^3/s]")

#rate_for = 1/sum([1/rates_for[react] for react in rates_for])
#rate_rev = 1/sum([1/rates_rev[react] for react in rates_rev])

#print(f"rate for {rate_for:+7.4e} [kmol/m^3/s]")
#print(f"rate rev {rate_rev:+7.4e} [kmol/m^3/s]")
#print(f"rate tot {rate_for+rate_rev:+7.4e} [kmol/m^3/s]")

print("\n ★ Reaction rates:")
for jj, react in enumerate(microkin.reactions):
    print(f"{react.name:50s} for {microkin.rates_for[jj]:+7.4e} [kmol/m^3/s]")
    print(f"{react.name:50s} rev {-microkin.rates_rev[jj]:+7.4e} [kmol/m^3/s]")
    print(f"{react.name:50s} tot {microkin.rates_net[jj]:+7.4e} [kmol/m^3/s]")

params = {
    'mathtext.default': 'regular',
}
plt.rcParams.update(params)

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
