# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
from ase.formula import Formula
from mikimoto import units
from mikimoto.utilities import NameAnalyzer

# -----------------------------------------------------------------------------
# KINETICS
# -----------------------------------------------------------------------------


class Species:
    
    def __init__(
        self,
        name,
        thermo,
        index = None,
        name_analyzer = NameAnalyzer(),
    ):
        self.name = name
        self.index = index
        self.thermo = thermo
        self.composition = name_analyzer.get_composition(self.name)
        self.size = name_analyzer.get_size(self.name)


class Reaction(Species):
    
    def __init__(
        self,
        name,
        thermo,
        index = None,
        name_analyzer = NameAnalyzer(),
    ):
        self.name = name
        self.index = index
        self.thermo = thermo
        self.reactants = name_analyzer.get_reactants(self.name)
        self.products = name_analyzer.get_products(self.name)
        self.composition = name_analyzer.get_composition(self.name)
        self.size = name_analyzer.get_size(self.name)
        name_analyzer.check_reaction(self.name)


class Solution:

    def __init__(
        self,
        species,
        reactions,
        temperature, # [K]
        pressure, # [Pa]
        pressure_ref = 1 * units.atm, # [Pa]
    ):
        self.species = species
        self.reactions = reactions
        self.pressure_ref = pressure_ref # [Pa]
        self.temperature = temperature # [K]
        self.pressure = pressure # [Pa]
        self.n_species = len(self.species)
        self.n_reactions = len(self.reactions)

    @property
    def temperature(self):
        return self._temperature # [K]

    @temperature.setter
    def temperature(self, temperature):
        for spec in self.species:
            spec.thermo.temperature = temperature
        self.conc_ref = self.pressure_ref/(units.Rgas*temperature)
        self._temperature = temperature

    @property
    def pressure(self):
        return self._pressure # [Pa]

    @pressure.setter
    def pressure(self, pressure):
        for spec in self.species:
            spec.thermo.pressure = pressure
        self._pressure = pressure

    @property
    def conc_ref(self):
        return self._conc_ref # [kmol/m^3]

    @conc_ref.setter
    def conc_ref(self, conc_ref):
        for spec in self.species:
            spec.thermo.conc_ref = conc_ref
        self._conc_ref = conc_ref


class Interface:

    def __init__(
        self,
        species,
        reactions,
        temperature, # [K]
        pressure, # [Pa]
        sites_conc, # [kmol/m^3]
    ):
        self.species = species
        self.reactions = reactions
        self.temperature = temperature # [K]
        self.pressure = pressure # [Pa]
        self.sites_conc = sites_conc # [kmol/m^3]
        self.n_species = len(self.species)
        self.n_reactions = len(self.reactions)

    @property
    def temperature(self):
        return self._temperature # [K]

    @temperature.setter
    def temperature(self, temperature):
        for spec in self.species:
            spec.thermo.temperature = temperature
        self._temperature = temperature

    @property
    def pressure(self):
        return self._pressure # [Pa]

    @pressure.setter
    def pressure(self, pressure):
        for spec in self.species:
            spec.thermo.pressure = pressure
        self._pressure = pressure

    @property
    def sites_conc(self):
        return self._sites_conc # [kmol/m^3]

    @sites_conc.setter
    def sites_conc(self, sites_conc):
        for spec in self.species:
            spec.thermo.conc_ref = sites_conc
        self._sites_conc = sites_conc


class Microkinetics:
    
    def __init__(
        self,
        temperature, # [K]
        pressure, # [Pa]
        gas_phases = [],
        surf_phases = [],
        log_rates_eval = False,
    ):
        self.temperature = temperature # [K]
        self.pressure = pressure # [Pa]
        self.gas_phases = gas_phases
        self.surf_phases = surf_phases
        self.log_rates_eval = log_rates_eval
        
        # Inizialize the model and calculate stoichiometric coefficients.
        self.initialize()
    
    def initialize(self):
    
        self.gas_species = []
        self.gas_reactions = []
        for phase in self.gas_phases:
            self.gas_species += phase.species
            self.gas_reactions += phase.reactions
        self.surf_species = []
        self.surf_reactions = []
        for phase in self.surf_phases:
            self.surf_species += phase.species
            self.surf_reactions += phase.reactions

        self.species = self.gas_species+self.surf_species
        self.reactions = self.gas_reactions+self.surf_reactions
        self.n_gas_species = len(self.gas_species)
        self.n_surf_species = len(self.surf_species)
        self.n_species = len(self.species)
        self.n_gas_reactions = len(self.gas_reactions)
        self.n_surf_reactions = len(self.surf_reactions)
        self.n_reactions = len(self.reactions)
        self.indices_gas = np.arange(0, self.n_gas_species)
        self.indices_surf = np.arange(self.n_gas_species, self.n_species)
        self.is_gas_spec = np.array(
            [1]*self.n_gas_species+[0]*self.n_surf_species
        )
    
        # Collect species_names and reactions_names lists.
        self.species_names = [spec.name for spec in self.species]
        self.reactions_names = [react.name for react in self.reactions]
        
        # Assign indexe to species and reactions.
        for ii, spec in enumerate(self.species_names):
            spec.index == ii
        for ii, react in enumerate(self.reactions_names):
            react.index == ii
        
        # Create dictionaries of indexes.
        self.species_indices = {
            spec: ii for ii, spec in enumerate(self.species_names)
        }
        self.reactions_indices = {
            react: ii for ii, react in enumerate(self.reactions_names)
        }
        
        # Create matrices of stoichiometric coefficients.
        # stoich_for and stoich_rev are the stoichiometric coefficients
        # of the reactants for forward and reverse reactions.
        self.stoich_for = np.zeros(
            shape = (self.n_reactions, self.n_species), dtype = int,
        )
        self.stoich_rev = np.zeros(
            shape = (self.n_reactions, self.n_species), dtype = int,
        )
        for rr, reaction in enumerate(self.reactions):
            for spec in reaction.reactants:
                ss = self.species_indices[spec]
                self.stoich_for[rr, ss] += 1
            for spec in reaction.products:
                ss = self.species_indices[spec]
                self.stoich_rev[rr, ss] += 1
        self.stoich_coeffs = -self.stoich_for+self.stoich_rev

    @property
    def conc_tot_gas(self):
        return self.pressure/units.Rgas/self.temperature

    @property
    def conc_tot_reactions(self):
        conc_tot = []
        for phase in self.gas_phases:
            conc_tot += [self.conc_tot_gas]*phase.n_reactions
        for phase in self.surf_phases:
            conc_tot += [phase.sites_conc]*phase.n_reactions
        return np.array(conc_tot) # [kmol/m^3]

    @property
    def conc_tot_species(self):
        conc_tot = []
        for phase in self.gas_phases:
            conc_tot += [self.conc_tot_gas]*phase.n_species
        for phase in self.surf_phases:
            conc_tot += [phase.sites_conc]*phase.n_species
        return np.array(conc_tot) # [kmol/m^3]

    def get_kinetic_constants(self):
        
        # Calculate activation energies (forward and reverse).
        spec_energies = np.array([spec.thermo.Gibbs_std for spec in self.species])
        ts_energies = np.array([react.thermo.Gibbs_std for react in self.reactions])
        act_energies_for = ts_energies-np.dot(self.stoich_for, spec_energies)
        act_energies_rev = ts_energies-np.dot(self.stoich_rev, spec_energies)

        # Calculate kinetic constants (forward and reverse).
        A_pre = units.kB*self.temperature/units.hP # [1/s]
        denom = units.Rgas*self.temperature
        self.kin_const_for = np.multiply(
            A_pre*np.exp(-act_energies_for/denom), self.conc_tot_reactions
        ) # [kmol/m^3/s]
        self.kin_const_rev = np.multiply(
            A_pre*np.exp(-act_energies_rev/denom), self.conc_tot_reactions
        ) # [kmol/m^3/s]

    def get_reaction_rates(self, conc, delta = 1e-50):
    
        # Calculate concentrations divided by total concentrations, i.e.,
        # molar fractions and surface coverages.
        conc_over_conc_tot = np.divide(conc, self.conc_tot_species)
    
        # Calcualate reaction rates (forward, reverse, and net).
        if self.log_rates_eval is True:
            # This is to prevent errors in the log.
            conc_over_conc_tot = np.abs(conc_over_conc_tot+delta)
            # Log method: prod(A^B) = exp(sum(B*log(A)).
            self.rates_for = self.kin_const_for*np.exp(np.dot(
                self.stoich_for, np.log(conc_over_conc_tot)
            ))
            self.rates_rev = self.kin_const_rev*np.exp(np.dot(
                self.stoich_rev, np.log(conc_over_conc_tot)
            ))
        else:
            self.rates_for = self.kin_const_for*np.prod(
                np.power(conc_over_conc_tot, self.stoich_for), axis = 1
            )
            self.rates_rev = self.kin_const_rev*np.prod(
                np.power(conc_over_conc_tot, self.stoich_rev), axis = 1
            )
        self.rates_net = self.rates_for-self.rates_rev # [kmol/m^3/s]


# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
