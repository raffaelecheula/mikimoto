# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
from ase.formula import Formula

# -----------------------------------------------------------------------------
# ENERGY REFERENCES
# -----------------------------------------------------------------------------


def get_composition_dict(species, name_analyzer=None):
    """Get a dictionary with the number of atoms for the elements of a species."""
    if name_analyzer:
        return name_analyzer.get_composition(species)
    else:
        return Formula(species, strict=True).count()


def get_elements_from_species_list(species, name_analyzer=None):
    """Get a list of elements from a list of species."""
    return list(get_composition_dict("".join(species), name_analyzer).keys())


def get_composition_matrix(species, elements, name_analyzer=None):
    """Get composition matrix for a set of species and elements."""
    elem_dict = {elem: ii for ii, elem in enumerate(elements)}
    if len(species) != len(elements):
        raise RuntimeError(
            "The number of species must be equal to the number of elements"
        )
    comp_matrix = np.zeros((len(species), len(elements)))
    for ii, spec in enumerate(species):
        comp_dict = get_composition_dict(spec, name_analyzer)
        for elem in comp_dict:
            jj = elem_dict[elem]
            comp_matrix[ii, jj] = comp_dict[elem]
    return comp_matrix


def get_energy_ref_dict(species, energies, print_energy_ref=False, name_analyzer=None):
    """Get the reference energies of a set of linearly independent species."""
    if isinstance(energies, dict):
        energies = [energies[spec] for spec in species]
    elements = get_elements_from_species_list(species, name_analyzer)
    comp_matrix = get_composition_matrix(species, elements)
    inv_matrix = np.linalg.inv(comp_matrix)
    energies_ref = np.dot(inv_matrix, energies)
    energy_ref_dict = {}
    for ii, elem in enumerate(elements):
        energy_ref_dict[elem] = energies_ref[ii]
    if print_energy_ref:
        print_energy_ref_formulas(elements, species, inv_matrix)
    return energy_ref_dict


def print_energy_ref_formulas(elements, species, inv_matrix):
    """Print formulas used to get the reference energies."""
    for ii, elem in enumerate(elements):
        print(f"E_{elem}: "+" ".join(
            [f"{inv_matrix[ii,jj]:+3.2f}*E_{spec}" for jj, spec in enumerate(species)]
        ))


def change_reference_energies(energy_dict, energy_ref_dict, name_analyzer=None):
    """Get an energy dictionary with the energy of the references set to zero."""
    energy_dict = energy_dict.copy()
    for spec in energy_dict:
        comp_dict = get_composition_dict(spec, name_analyzer)
        for elem in comp_dict:
            energy_dict[spec] -= energy_ref_dict[elem] * comp_dict[elem]
    return energy_dict


def get_energy_corrections(
    energy_dict_right,
    energy_dict_wrong,
    species,
    name_analyzer=None,
):
    """Get the energy corrections to apply to energy_dict_wrong (e.g., from DFT
    energies) to get the same reaction enthalpies that energy_dict_right gives 
    (e.g., from NIST-Janaf tables or NASA coefficients)."""
    energy_ref_dict = get_energy_ref_dict(
        species = species,
        energies = energy_dict_right,
    )
    energy_dict_right = change_reference_energies(
        energy_dict = energy_dict_right,
        energy_ref_dict = energy_ref_dict,
        name_analyzer = name_analyzer,
    )

    energy_ref_dict = get_energy_ref_dict(
        species = species,
        energies = energy_dict_wrong,
    )
    energy_dict_wrong = change_reference_energies(
        energy_dict = energy_dict_wrong,
        energy_ref_dict = energy_ref_dict,
        name_analyzer = name_analyzer,
    )
    energy_dict_corr = {}
    for spec in energy_dict_right:
        energy_dict_corr[spec] = energy_dict_right[spec]-energy_dict_wrong[spec]
    
    return energy_dict_corr

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
