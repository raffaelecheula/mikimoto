# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
from ase.formula import Formula
from ase.data import atomic_masses, atomic_numbers
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
        name_analyzer = NameAnalyzer(),
    ):
        self.name = name
        self.thermo = thermo
        self.composition = name_analyzer.get_composition(self.name)
        self.size = name_analyzer.get_size(self.name)
        self.molar_mass = name_analyzer.get_molar_mass(self.name) # [kg/kmol]

    @property
    def conc(self):
        """Molar concentration of the species, in [kmol/m^3]."""
        return self._conc # [kmol/m^3]

    @conc.setter
    def conc(self, conc):
        self._conc = conc # [kmol/m^3]
        self._rho = conc*self.molar_mass # [kg/m^3]

    @property
    def rho(self):
        """Mass concentration of the species, in [kg/m^3]."""
        return self._rho # [kg/m^3]

    @rho.setter
    def rho(self, rho):
        self._rho = rho # [kg/m^3]
        self._conc = rho/self.molar_mass # [kmol/m^3]


class Reaction:
    
    def __init__(
        self,
        name,
        thermo,
        name_analyzer = NameAnalyzer(),
    ):
        self.name = name
        self.thermo = thermo
        self.reactants = name_analyzer.get_reactants(self.name)
        self.products = name_analyzer.get_products(self.name)
        self.composition = name_analyzer.get_composition(self.name)
        self.size = name_analyzer.get_size(self.name)
        self.gas_ads_dict = name_analyzer.get_gas_ads_species_dict(self.name)
        self.gas_species = name_analyzer.get_gas_species(self.name)
        self.ads_species = name_analyzer.get_ads_species(self.name)
        name_analyzer.check_reaction(self.name)


class Phase:

    def __init__(
        self,
        species,
        reactions,
        temperature, # [K]
        pressure, # [Pa]
    ):
        if isinstance(species, dict):
            species = list(species.values())
        if isinstance(reactions, dict):
            reactions = list(reactions.values())
        self.species = species
        self.reactions = reactions
        self.temperature = temperature # [K]
        self.pressure = pressure # [Pa]
        self.n_species = len(self.species)
        self.n_reactions = len(self.reactions)

    @property
    def temperature(self):
        """Tepmerature of the phase in [K]."""
        return self._temperature # [K]

    @temperature.setter
    def temperature(self, temperature):
        for spec in self.species+self.reactions:
            spec.thermo.temperature = temperature # [K]
        self._temperature = temperature # [K]

    @property
    def pressure(self):
        """Pressure of the phase, in [Pa]."""
        return self._pressure # [Pa]

    @pressure.setter
    def pressure(self, pressure):
        for spec in self.species+self.reactions:
            spec.thermo.pressure = pressure # [Pa]
        self._pressure = pressure # [Pa]

    @property
    def conc_ref(self):
        """Reference molar concentration of the phase, in [kmol/m^3]."""
        return self._conc_ref # [kmol/m^3]

    @conc_ref.setter
    def conc_ref(self, conc_ref):
        for spec in self.species+self.reactions:
            spec.thermo.conc_ref = conc_ref # [kmol/m^3]
        self._conc_ref = conc_ref # [kmol/m^3]

    @property
    def conc_list(self):
        """List of molar concentrations of species, in [kmol/m^3]."""
        return [spec.conc for spec in self.species] # [kmol/m^3]

    @conc_list.setter
    def conc_list(self, conc_list):
        for ii, spec in enumerate(self.species):
            spec.conc = conc_list[ii] # [kmol/m^3]
        self._conc_tot = np.sum(conc_list)
        self._rho_tot = self.conc_tot*self.molar_mass # [kg/m^3]

    @property
    def rho_list(self):
        """List of mass concentrations of species, in [kg/m^3]."""
        return [spec.rho for spec in self.species] # [kg/m^3]

    @rho_list.setter
    def rho_list(self, rho_list):
        for ii, spec in enumerate(self.species):
            spec.rho = rho_list[ii] # [kg/m^3]
        self._rho_tot = np.sum(rho_list)
        self._conc_tot = self.rho_tot*self.reciprocal_molar_mass # [kmol/m^3]

    @property
    def conc_tot(self):
        """Total molar concentration of the phase, in [kmol/m^3]."""
        return self._conc_tot # [kmol/m^3]

    @conc_tot.setter
    def conc_tot(self, conc_tot):
        self._conc_tot = conc_tot # [kmol/m^3]
        self._rho_tot = conc_tot*self.molar_mass # [kg/m^3]

    @property
    def rho_tot(self):
        """Total mass concentration of the phase, in [kg/m^3]."""
        return self._rho_tot # [kg/m^3]

    @rho_tot.setter
    def rho_tot(self, rho_tot):
        self._rho_tot = rho_tot # [kg/m^3]
        self._conc_tot = rho_tot*self.reciprocal_molar_mass # [kmol/m^3]

    @property
    def X_list(self):
        """List of molar fractions of species, in [kmol/kmol]."""
        return [spec.conc/self.conc_tot for spec in self.species] # [-]

    @X_list.setter
    def X_list(self, X_list):
        for ii, spec in enumerate(self.species):
            spec.conc = X_list[ii]*self.conc_tot # [-]
        self._rho_tot = self.conc_tot*self.molar_mass # [kg/m^3]

    @property
    def X_dict(self):
        """Dictionary of molar fractions of species, in [kmol/kmol]."""
        return {spec.name: spec.conc/self.conc_tot for spec in self.species} # [-]

    @X_dict.setter
    def X_dict(self, X_dict):
        for spec in self.species:
            spec.conc = X_dict[spec.name]*self.conc_tot # [-]
        self._rho_tot = self.conc_tot*self.molar_mass # [kg/m^3]

    @property
    def Y_list(self):
        """List of mass fractions of species, in [kg/kg]."""
        return [spec.rho/self.rho_tot for spec in self.species] # [-]

    @Y_list.setter
    def Y_list(self, Y_list):
        for ii, spec in enumerate(self.species):
            spec.rho = Y_list[ii]*self.rho_tot # [-]
        self._conc_tot = self.rho_tot*self.reciprocal_molar_mass # [kmol/m^3]

    @property
    def Y_dict(self):
        """Dictionary of mass fractions of species, in [kg/kg]."""
        return {spec.name: spec.rho/self.rho_tot for spec in self.species} # [-]

    @Y_dict.setter
    def Y_dict(self, Y_dict):
        for spec in self.species:
            spec.rho = Y_dict[spec.name]*self.rho_tot # [-]
        self._conc_tot = self.rho_tot*self.reciprocal_molar_mass # [kmol/m^3]

    @property
    def molar_mass_list(self):
        """List of molar masses of the species, in [kg/kmol]."""
        return [spec.molar_mass for spec in self.species] # [kg/kmol]

    @property
    def molar_mass(self):
        """Molar mass (molecular weight) of the phase, in [kg/kmol]."""
        return sum(
            [spec.conc/self.conc_tot*spec.molar_mass for spec in self.species]
        ) # [kg/kmol]

    @property
    def reciprocal_molar_mass(self):
        """Reciprocal molar mass (1 / molar mass) of the phase, in [kmol/kg]."""
        return sum(
            [spec.rho/self.rho_tot/spec.molar_mass for spec in self.species]
        ) # [kmol/kg]


class IdealGas(Phase):

    def __init__(
        self,
        species,
        reactions,
        temperature, # [K]
        pressure, # [Pa]
        pressure_ref = 1 * units.atm, # [Pa]
    ):
        self._temperature = temperature # [K]
        self._pressure = pressure # [Pa]
        self.pressure_ref = pressure_ref # [Pa]
        super().__init__(
            species = species,
            reactions = reactions,
            temperature = temperature,
            pressure = pressure,
        )

    @Phase.temperature.setter
    def temperature(self, temperature):
        for spec in self.species+self.reactions:
            spec.thermo.temperature = temperature # [K]
        self._conc_tot = self.pressure/(units.Rgas*temperature) # [kmol/m^3]
        self.conc_ref = self.pressure_ref/(units.Rgas*temperature) # [kmol/m^3]
        self._temperature = temperature # [K]

    @Phase.pressure.setter
    def pressure(self, pressure):
        for spec in self.species+self.reactions:
            spec.thermo.pressure = pressure # [Pa]
        self._conc_tot = self.pressure/(units.Rgas*self.temperature) # [kmol/m^3]
        self._pressure = pressure # [Pa]


class Interface(Phase):

    def __init__(
        self,
        species,
        reactions,
        temperature, # [K]
        pressure, # [Pa]
        conc_tot, # [kmol/m^3]
    ):
        super().__init__(
            species = species,
            reactions = reactions,
            temperature = temperature,
            pressure = pressure,
        )
        self.conc_tot = conc_tot # [kmol/m^3]

    @property
    def conc_tot(self):
        """Total molar concentration of the phase, in [kmol/m^3]."""
        return self._conc_tot # [kmol/m^3]

    @conc_tot.setter
    def conc_tot(self, conc_tot):
        for spec in self.species+self.reactions:
            spec.thermo.conc_ref = conc_tot # [kmol/m^3]
        self._conc_tot = conc_tot # [kmol/m^3]

    @property
    def conc_ref(self):
        """Reference molar concentration of the phase, in [kmol/m^3]."""
        return self._conc_tot # [kmol/m^3]

    @property
    def rho_tot(self):
        """Total mass concentration of the phase, in [kg/m^3]."""
        return self.conc_tot*self.molar_mass # [kg/m^3]

    @property
    def θ_list(self):
        """List of coverages of the phase, in [kmol/kmol]."""
        return self.X_list # [-]

    @θ_list.setter
    def θ_list(self, θ_list):
        self.X_list = θ_list # [-]

    @property
    def θ_dict(self):
        """Dictionary of coverages of the phase, in [kmol/kmol]."""
        return self.X_dict # [-]

    @θ_dict.setter
    def θ_dict(self, θ_dict):
        self.X_dict = θ_dict # [-]


class Microkinetics:
    
    def __init__(
        self,
        temperature, # [K]
        pressure, # [Pa]
        gas_phases = [],
        surf_phases = [],
        units_energy = units.J/units.kmol, # [J/kmol]
        log_rates_eval = False,
    ):
        self.gas_phases = gas_phases
        self.surf_phases = surf_phases
        self.phases = self.gas_phases+self.surf_phases
        self.temperature = temperature # [K]
        self.pressure = pressure # [Pa]
        self.units_energy = units_energy # [J/kmol]
        self.log_rates_eval = log_rates_eval
        
        # Inizialize the model and calculate stoichiometric coefficients.
        self.initialize()
    
    def initialize(self):
        
        # Information on gas phases.
        self.gas_species = []
        self.gas_reactions = []
        for phase in self.gas_phases:
            self.gas_species += phase.species
            self.gas_reactions += phase.reactions
        self.n_gas_species = len(self.gas_species)
        self.n_gas_reactions = len(self.gas_reactions)
        
        # Information on surface phases.
        self.surf_species = []
        self.surf_reactions = []
        for phase in self.surf_phases:
            self.surf_species += phase.species
            self.surf_reactions += phase.reactions
        self.n_surf_species = len(self.surf_species)
        self.n_surf_reactions = len(self.surf_reactions)

        # Information on all phases.
        self.species = self.gas_species+self.surf_species
        self.reactions = self.gas_reactions+self.surf_reactions
        self.n_species = len(self.species)
        self.n_reactions = len(self.reactions)
        
        # Delimiters to split lists of quantities into corresponding phases.
        i_species = 0
        i_reactions = 0
        self.delimiter_species = []
        self.delimiter_reactions = []
        for phase in self.phases[:-1]:
            i_species += phase.n_species
            i_reactions += phase.n_reactions
            self.delimiter_species.append(i_species)
            self.delimiter_reactions.append(i_reactions)
        i_species = 0
        i_reactions = 0
        self.delimiter_gas_species = []
        self.delimiter_gas_reactions = []
        for phase in self.gas_phases[:-1]:
            i_species += phase.n_species
            i_reactions += phase.n_reactions
            self.delimiter_gas_species.append(i_species)
            self.delimiter_gas_reactions.append(i_reactions)
        i_species = 0
        i_reactions = 0
        self.delimiter_surf_species = []
        self.delimiter_surf_reactions = []
        for phase in self.surf_phases[:-1]:
            i_species += phase.n_species
            i_reactions += phase.n_reactions
            self.delimiter_surf_species.append(i_species)
            self.delimiter_surf_reactions.append(i_reactions)
        
        # Indices of gas and surface phases.
        self.indices_gas = np.arange(0, self.n_gas_species)
        self.indices_surf = np.arange(self.n_gas_species, self.n_species)
        self.is_gas = np.array([1]*self.n_gas_species+[0]*self.n_surf_species)
    
        # Collect species_names and reactions_names lists.
        self.species_names = [spec.name for spec in self.species]
        self.reactions_names = [react.name for react in self.reactions]
        
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

    def get_kinetic_constants(self):
        
        # Calculate activation energies (forward and reverse).
        spec_energies = np.array([spec.thermo.Gibbs_std for spec in self.species])
        ts_energies = np.array([react.thermo.Gibbs_std for react in self.reactions])
        act_energies_for = ts_energies-np.dot(self.stoich_for, spec_energies)
        act_energies_rev = ts_energies-np.dot(self.stoich_rev, spec_energies)

        # Calculate kinetic constants (forward and reverse).
        A_pre = units.kB*self.temperature/units.hP # [1/s]
        denom = units.Rgas*self.temperature/self.units_energy # [J/kmol]
        self.kin_const_for = np.multiply(
            A_pre*np.exp(-act_energies_for/denom), self.conc_tot_reactions
        ) # [kmol/m^3/s]
        self.kin_const_rev = np.multiply(
            A_pre*np.exp(-act_energies_rev/denom), self.conc_tot_reactions
        ) # [kmol/m^3/s]

    def get_reaction_rates(self, conc = None, delta = 1e-50):
    
        # Calculate concentrations divided by reference concentrations.
        if conc is None:
            conc = self.conc_list # [kmol/m^3]
        conc_over_conc_ref = np.divide(conc, self.conc_ref_species) # [-]
    
        # Calcualate reaction rates (forward, reverse, and net).
        if self.log_rates_eval is True:
            # This is to prevent errors in the log.
            conc_over_conc_ref = np.abs(conc_over_conc_ref+delta)
            # Log method: prod(A^B) = exp(sum(B*log(A)).
            self.rates_for = self.kin_const_for*np.exp(np.dot(
                self.stoich_for, np.log(conc_over_conc_ref)
            )) # [kmol/m^3/s]
            self.rates_rev = self.kin_const_rev*np.exp(np.dot(
                self.stoich_rev, np.log(conc_over_conc_ref)
            )) # [kmol/m^3/s]
        else:
            self.rates_for = self.kin_const_for*np.prod(
                np.power(conc_over_conc_ref, self.stoich_for), axis = 1
            ) # [kmol/m^3/s]
            self.rates_rev = self.kin_const_rev*np.prod(
                np.power(conc_over_conc_ref, self.stoich_rev), axis = 1
            ) # [kmol/m^3/s]
        self.rates_net = self.rates_for-self.rates_rev # [kmol/m^3/s]

    def get_net_production_rates(self):
        """Get the net production rates of the species, in [kmol/m^3/s]."""
        return np.dot(self.stoich_coeffs.T, self.rates_net) # [kmol/m^3/s]

    def get_creation_rates(self):
        """Get the forward creation rates of the species, in [kmol/m^3/s]."""
        return np.dot(self.stoich_for.T, self.rates_for) # [kmol/m^3/s]

    def get_consumption_rates(self):
        """Get the consumption rates of the species, in [kmol/m^3/s]."""
        return np.dot(self.stoich_rev.T, self.rates_rev) # [kmol/m^3/s]

    @property
    def temperature(self):
        """Temperature of the simulation, in [K]."""
        return self._temperature # [K]

    @temperature.setter
    def temperature(self, temperature):
        for phase in self.phases:
            phase.temperature = temperature # [K]
        self._temperature = temperature # [K]

    @property
    def pressure(self):
        """Pressure of the simulation, in [Pa]."""
        return self._pressure # [Pa]

    @pressure.setter
    def pressure(self, pressure):
        for phase in self.phases:
            phase.pressure = pressure # [Pa]
        self._pressure = pressure # [Pa]

    @property
    def conc_tot_reactions(self):
        """Array of total concentrations for the reactions, in [kmol/m^3]"""
        conc_tot = []
        for phase in self.phases:
            conc_tot += [phase.conc_tot]*phase.n_reactions
        return np.array(conc_tot) # [kmol/m^3]

    @property
    def conc_ref_species(self):
        """Array of reference concentrations for the species, in [kmol/m^3]"""
        conc_tot = []
        for phase in self.phases:
            conc_tot += [phase.conc_ref]*phase.n_species
        return np.array(conc_tot) # [kmol/m^3]

    @property
    def conc_list(self):
        """List of molar concentrations of the species, in [kmol/m^3]."""
        conc_list = []
        for phase in self.phases:
            conc_list += phase.conc_list
        return conc_list # [kmol/m^3]

    @conc_list.setter
    def conc_list(self, conc_list):
        conc_list_list = np.split(conc_list, self.delimiter_species)
        for ii, phase in enumerate(self.phases):
            phase.conc_list = conc_list_list[ii] # [kmol/m^3]

    @property
    def rho_list(self):
        """List of mass concentrations of the species, in [kmol/m^3]."""
        rho_list = []
        for phase in self.phases:
            rho_list += phase.rho_list
        return rho_list # [kg/m^3]

    @property
    def molar_mass_list(self):
        """List of molar masses of the species, in [kg/kmol]."""
        molar_mass_list = []
        for phase in self.phases:
            molar_mass_list += phase.molar_mass_list
        return molar_mass_list # [kg/kmol]

    @rho_list.setter
    def rho_list(self, rho_list):
        rho_list_list = np.split(rho_list, self.delimiter_species)
        for ii, phase in enumerate(self.phases):
            phase.rho_list = rho_list_list[ii] # [kmol/m^3]

    @property
    def X_gas_list(self):
        """List of molar fractions of the gas species, in [kmol/kmol]."""
        X_list = []
        for phase in self.gas_phases:
            X_list += phase.X_list
        return X_list # [-]

    @X_gas_list.setter
    def X_gas_list(self, X_gas_list):
        X_list_list = np.split(X_gas_list, self.delimiter_gas_species)
        for ii, phase in enumerate(self.gas_phases):
            phase.X_list = X_list_list[ii]

    @property
    def θ_surf_list(self):
        """List of coverages of the surface species, in [kg/kg]."""
        θ_list = []
        for phase in self.surf_phases:
            θ_list += phase.θ_list
        return θ_list # [-]

    @θ_surf_list.setter
    def θ_surf_list(self, θ_surf_list):
        θ_list_list = np.split(θ_surf_list, self.delimiter_surf_species)
        for ii, phase in enumerate(self.surf_phases):
            phase.θ_list = θ_list_list[ii]

    @property
    def conc_gas_tot(self):
        """Total molar concentration of gas phases, in [kmol/m^3]."""
        return sum([phase.conc_tot for phase in self.gas_phases]) # [kmol/m^3]

    @property
    def rho_gas_tot(self):
        """Total mass concentration of gas phases, in [kg/m^3]."""
        return sum([phase.rho_tot for phase in self.gas_phases]) # [kg/m^3]

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
