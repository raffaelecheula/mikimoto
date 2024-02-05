# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
from ase.formula import Formula
from ase.data import atomic_masses, atomic_numbers

# -----------------------------------------------------------------------------
# COMPOSITION ANALYSIS
# -----------------------------------------------------------------------------


def get_composition_dict(species, name_analyzer=None):
    """Get a dictionary with the number of atoms for the elements of a species."""
    from mikimoto.microkinetics import Species
    if isinstance(species, Species):
        species = species.name
    if name_analyzer:
        return name_analyzer.get_composition(species)
    else:
        return Formula(species, strict=True).count()


def get_elements_from_species_list(species, name_analyzer=None):
    """Get a list of elements from a list of species."""
    return list(get_composition_dict("".join(species), name_analyzer).keys())


def get_molecular_mass(species, name_analyzer=None):
    """Get the molecular mass of a species."""
    comp_dict = get_composition_dict(species=species, name_analyzer=name_analyzer)
    return sum([atomic_masses[atomic_numbers[ii]]*comp_dict[ii] for ii in comp_dict])


# -----------------------------------------------------------------------------
# NAME ANALYZER
# -----------------------------------------------------------------------------


class NameAnalyzer():

    def __init__(
        self,
        site_separators = "(,)",
        react_separator = "<=>",
        text_separator = "_",
        ignore_characters = "",
    ):
        self.site_separators = site_separators
        self.react_separator = react_separator
        self.text_separator = text_separator
        self.ignore_characters = ignore_characters

    def check_name(self, name):
        """Check unwanted additional spaces in the name."""
        if name[0] == " " or name[-1] == " " or "  " in name:
            raise RuntimeError(f'Error in: {name}. Check unwanted spaces.')

    def get_names_reactants_products(self, name):
        """Split equation into names of reactants and products."""
        if self.react_separator in name:
            names = name.split(f" {self.react_separator} ")
        else:
            raise RuntimeError(
                f"The reaction {name} does not contain {self.react_separator}."
            )
        for name in names:
            self.check_name(name)
        return names

    def get_name_without_mult_integers(self, name):
        """Convert multiplying integers into the corresponding pieces."""
        self.check_name(name)
        pieces = []
        for piece in name.split(" + "):
            if " " in piece:
                mm, pp = piece.split(" ")
                pieces += [pp]*int(mm)
            else:
                pieces += [piece]
        return " + ".join(pieces)

    def get_reactants(self, name):
        """Get a list of reactants of the reaction."""
        name = self.get_names_reactants_products(name=name)[0]
        name = self.get_name_without_mult_integers(name=name)
        return name.split(" + ")

    def get_products(self, name):
        """Get a list of products of the reaction."""
        name = self.get_names_reactants_products(name=name)[1]
        name = self.get_name_without_mult_integers(name=name)
        return name.split(" + ")

    def get_gas_ads_species_dict(self, name):
        """Get a dictionary of gas species and adsorbed species."""
        gas_ads_dict = {}
        reactants = self.get_reactants(name)
        gas_ads_dict['reactants_gas'] = [
            spec for spec in reactants if self.site_separators[0] not in spec
        ]
        gas_ads_dict['reactants_ads'] = [
            spec for spec in reactants if self.site_separators[0] in spec
        ]
        
        products = self.get_products(name)
        gas_ads_dict['products_gas'] = [
            spec for spec in products if self.site_separators[0] not in spec
        ]
        gas_ads_dict['products_ads'] = [
            spec for spec in products if self.site_separators[0] in spec
        ]
        return gas_ads_dict

    def get_gas_species(self, name):
        """Get all the gas species in the name."""
        gas_ads_dict = self.get_gas_ads_species_dict(name)
        return gas_ads_dict['reactants_gas']+gas_ads_dict['products_gas']

    def get_ads_species(self, name):
        """Get all the ads species in the name."""
        gas_ads_dict = self.get_gas_ads_species_dict(name)
        return gas_ads_dict['reactants_ads']+gas_ads_dict['products_ads']

    def get_composition(self, name, index = 0):
        """Get composition of a species."""
        if self.react_separator in name:
            name = self.get_names_reactants_products(name=name)[index]
        name = self.get_name_without_mult_integers(name=name)
        name = name.replace(" ", "")
        for sep in self.site_separators:
            name = name.replace(sep, "+")
        name = "".join(
            [piece.split(self.text_separator)[0] for piece in name.split("+")]
        )
        for char in self.ignore_characters:
            name = name.replace(char, "")
        name = name.replace("+", "")
        return Formula(name, strict=True).count()

    def get_size(self, name, index = 0):
        """Get composition of a species."""
        if self.react_separator in name:
            name = self.get_names_reactants_products(name=name)[index]
        name = self.get_name_without_mult_integers(name=name)
        return name.count(self.site_separators[0])+name.count(self.site_separators[1])

    def get_composition_and_size(self, name, index = 0):
        """Get composition and size of a species."""
        composition = self.get_composition(name, index=index)
        size = self.get_size(name, index=index)
        return composition, size

    def check_reaction(self, name):
        """Check unwanted additional spaces in the name."""
        if self.get_composition(name, index=0) != self.get_composition(name, index=1):
            raise RuntimeError(
                f'Error in: {name}. Elements in reactants and products do not match.'
            )


# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
