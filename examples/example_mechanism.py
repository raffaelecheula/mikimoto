# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
from mikimoto import units
from mikimoto.energy_references import (
    get_energy_corrections,
    get_energy_ref_dict,
    change_reference_energies,
)

# -----------------------------------------------------------------------------
# PARAMETERS
# -----------------------------------------------------------------------------

# Read NASA coefficients of gas phase molecules.

# Read DFT gas phase molecules.

# Read DFT clean slab structure.

# Read DFT adsorbates structures.

# Read DFT transition state structures.

# Calculate correction to energies of DFT gas phase molecules.

# These are the energies (at 0 K) from experimental data (e.g., NIST-Janaf or
# NASA coefficients).
energy_dict_NASA = {
    "O2": -0.087,
    "H2": -0.087,
    "H2O": -2.611,
    "CO2": -4.168,
    "CO": -1.236,
    "H2CO": -1.320,
    "CH3OH": -2.187,
    "CH4": -0.883,
}

# These are energies + ZPE (at 0 K) obtained with DFT calculations.
energy_dict_DFT = {
    "O2": -1129.710,
    "H2": -31.459,
    "H2O": -598.572,
    "CO2": -1384.338,
    "CO": -816.352,
    "H2CO": -848.082,
    "CH3OH": -880.442,
    "CH4": -314.492,
}

# These are the species that are not affected by the uncertainty in the
# DFT calculations.
species_right = ['H2', 'H2O', 'CH4']

energy_dict_corr = get_energy_corrections(
    energy_dict_right = energy_dict_NASA,
    energy_dict_wrong = energy_dict_DFT,
    species = species_right,
)

energy_dict_DFT = {
    spec: energy_dict_DFT[spec]+energy_dict_corr[spec] 
    for spec in energy_dict_DFT
}
#print(energy_dict_DFT)

# These are the species that we want to use as reference for the calculation
# of the energies.
species_ref = ['H2', 'CO2', 'CO']

energy_ref_dict = get_energy_ref_dict(
    species = species_ref,
    energies = energy_dict_DFT,
    print_energy_ref = False,
    name_analyzer = None,
)
energy_dict = change_reference_energies(
    energy_dict = energy_dict_DFT,
    energy_ref_dict = energy_ref_dict,
    name_analyzer = None,
)
#print(energy_dict)

########
# TESTS:
########

sites_area = 105.945/9 * units.Ang**2 # [m^2]
sites_surf_conc = 1/sites_area/units.Navo # [kmol/m^2]
temperature = 320.+273.15
pressure_ref = 1 * units.atm

from mikimoto.microkinetics import Species
from mikimoto.thermodynamics import ThermoCanteraConstantCp, Thermo2Dgas
from mikimoto.utilities import get_molecular_mass

species_name = 'H2'
thermo = ThermoCanteraConstantCp(
    enthalpy_ref=0.,
    entropy_ref=0.,
    specific_heat_ref=0.,
    temperature_ref=320.+273.15,
)
species = Species(name=species_name, thermo=thermo)

thermo = Thermo2Dgas(
    species = species,
    sites_surf_conc = sites_surf_conc,
    temperature = temperature,
)

print(-thermo.entropy_std*temperature / (units.eV/units.molecule))

molecular_mass = get_molecular_mass(species=species_name)
k_for = (
    pressure_ref/sites_surf_conc/np.sqrt(2*np.pi*molecular_mass*units.Rgas*temperature)
) # [1/s]
delta_entropy = units.Rgas * np.log(k_for*units.hP/units.kB/temperature)

print(-delta_entropy*temperature / (units.eV/units.molecule))

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
