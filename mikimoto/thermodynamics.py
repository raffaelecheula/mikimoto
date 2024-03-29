# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import copy as cp
from mikimoto import units
from mikimoto.microkinetics import Species
from mikimoto.utilities import get_molar_mass

# -----------------------------------------------------------------------------
# THERMO
# -----------------------------------------------------------------------------


class Thermo:

    def __init__(
        self,
        temperature = None, # [K]
        conc = None, # [kmol/m^3]
        conc_ref = None, # [kmol/m^3]
    ):
        self.temperature = temperature # [K]
        self.conc = conc # [kmol/m^3]
        self.conc_ref = conc_ref # [kmol/m^3]

    @property
    def temperature(self):
        """Temperature of the species."""
        if self._temperature is None:
            raise RuntimeError("temperature not set.")
        return self._temperature # [K]

    @temperature.setter
    def temperature(self, temperature):
        self._temperature = temperature # [K]

    @property
    def conc(self):
        """Molar concentration of the species."""
        if self._conc is None:
            raise RuntimeError("conc not set.")
        return self._conc # [kmol/m^3]

    @conc.setter
    def conc(self, conc):
        self._conc = conc # [kmol/m^3]

    @property
    def conc_ref(self):
        """Reference molar concentration of the species."""
        if self._conc_ref is None:
            raise RuntimeError("conc_ref not set.")
        return self._conc_ref # [kmol/m^3]

    @conc_ref.setter
    def conc_ref(self, conc_ref):
        self._conc_ref = conc_ref # [kmol/m^3]

    @property
    def enthalpy(self):
        """Enthalpy of the species."""
        raise RuntimeError("No enthalpy available.")

    @property
    def entropy_std(self):
        """Standard entropy of the species."""
        raise RuntimeError("No entropy_std available.")

    @property
    def specific_heat(self):
        """Specific heat of the species."""
        raise RuntimeError("No specific_heat available.")

    @property
    def Gibbs_std(self):
        """Standard Gibbs free energy of the species."""
        return self.enthalpy - self.temperature * self.entropy_std # [J/kmol]

    @property
    def entropy(self):
        """Entropy of the species."""
        X_spec = self.conc/self.conc_ref
        log_X_spec = np.log(X_spec) if X_spec > 0. else -np.inf
        return self.entropy_std + units.Rgas*self.temperature*log_X_spec # [J/kmol/K]

    @property
    def Gibbs(self):
        """Gibbs free energy of the species."""
        return self.enthalpy - self.temperature*self.entropy # [J/kmol]

    def modify_energy(self, delta_energy):
        pass

    def copy(self):
        return cp.deepcopy(self)


class ThermoFixedG0(Thermo):

    def __init__(
        self,
        Gibbs_ref, # [J/kmol]
        pressure_ref = 1 * units.atm, # [Pa]
        temperature = None, # [K]
        conc = None, # [kmol/m^3]
        conc_ref = None, # [kmol/m^3]
    ):
        self.Gibbs_ref = Gibbs_ref # [J/kmol]
        self.pressure_ref = pressure_ref
        super().__init__(
            temperature = temperature,
            conc = conc,
            conc_ref = conc_ref,
        )

    @property
    def enthalpy(self):
        raise RuntimeError("No enthalpy available in ThermoFixedG0.")

    @property
    def entropy_std(self):
        raise RuntimeError("No entropy_std available in ThermoFixedG0.")

    @property
    def specific_heat(self):
        raise RuntimeError("No specific_heat available in ThermoFixedG0.")

    @property
    def Gibbs_std(self):
        return self.Gibbs_ref # [J/kmol]

    @property
    def entropy(self):
        raise RuntimeError("No entropy available in ThermoFixedG0.")

    @property
    def Gibbs(self):
        X_spec = self.conc/self.conc_ref # [-]
        log_X_spec = np.log(X_spec) if X_spec > 0. else -np.inf
        return self.Gibbs_std + units.Rgas*self.temperature*log_X_spec # [J/kmol]

    def modify_energy(self, delta_energy):
        self.Gibbs_ref += delta_energy # [J/kmol]


class ThermoConstantCp(Thermo):

    def __init__(
        self,
        enthalpy_ref, # [J/kmol]
        entropy_ref, # [J/kmol/K]
        specific_heat_ref, # [J/kmol/K]
        temperature_ref, # [K]
        pressure_ref = 1 * units.atm, # [Pa]
        temperature = None, # [K]
        conc = None, # [kmol/m^3]
        conc_ref = None, # [kmol/m^3]
    ):
        self.enthalpy_ref = enthalpy_ref # [J/kmol]
        self.entropy_ref = entropy_ref # [J/kmol/K]
        self.specific_heat_ref = specific_heat_ref # [J/kmol/K]
        self.temperature_ref = temperature_ref # [K]
        self.pressure_ref = pressure_ref # [Pa]
        super().__init__(
            temperature = temperature,
            conc = conc,
            conc_ref = conc_ref,
        )

    @property
    def enthalpy(self):
        return self.enthalpy_ref + self.specific_heat_ref * (
            self.temperature - self.temperature_ref
        ) # [J/kmol]

    @property
    def entropy_std(self):
        return self.entropy_ref + self.specific_heat_ref * np.log(
            self.temperature / self.temperature_ref
        ) # [J/kmol/K]

    @property
    def specific_heat(self):
        return self.specific_heat_ref # [J/kmol/K]

    def modify_energy(self, delta_energy):
        self.enthalpy_ref += delta_energy # [J/kmol]


class ThermoNASA7(Thermo):

    def __init__(
        self,
        coeffs_NASA,
        pressure_ref = 1 * units.atm, # [Pa]
        temperature = None, # [K]
        conc = None, # [kmol/m^3]
        conc_ref = None, # [kmol/m^3]
    ):
        if isinstance(coeffs_NASA, dict):
            coeffs_NASA = list(coeffs_NASA.values())
        self.coeffs_NASA = coeffs_NASA
        self.pressure_ref = pressure_ref # [Pa]
        super().__init__(
            temperature = temperature,
            conc = conc,
            conc_ref = conc_ref,
        )

    @property
    def enthalpy(self):
        return units.Rgas * (
            self.coeffs_NASA[0] * self.temperature + 
            self.coeffs_NASA[1]/2 * self.temperature**2 + 
            self.coeffs_NASA[2]/3 * self.temperature**3 + 
            self.coeffs_NASA[3]/4 * self.temperature**4 + 
            self.coeffs_NASA[4]/5 * self.temperature**5 + 
            self.coeffs_NASA[5]
        ) # [J/kmol]

    @property
    def entropy_std(self):
        return units.Rgas * (
            self.coeffs_NASA[0] * np.log(self.temperature) + 
            self.coeffs_NASA[1] * self.temperature + 
            self.coeffs_NASA[2]/2 * self.temperature**2 + 
            self.coeffs_NASA[3]/3 * self.temperature**3 + 
            self.coeffs_NASA[4]/4 * self.temperature**4 + 
            self.coeffs_NASA[6]
        ) # [J/kmol/K]

    @property
    def specific_heat(self):
        return units.Rgas * (
            self.coeffs_NASA[0] + 
            self.coeffs_NASA[1] * self.temperature + 
            self.coeffs_NASA[2] * self.temperature**2 + 
            self.coeffs_NASA[3] * self.temperature**3 + 
            self.coeffs_NASA[4] * self.temperature**4
        ) # [J/kmol/K]

    def modify_energy(self, delta_energy):
        self.coeffs_NASA[5] += delta_energy / units.Rgas # [J/kmol]


class ThermoShomate(Thermo):

    def __init__(
        self,
        coeffs_Shomate,
        pressure_ref = 1 * units.atm, # [Pa]
        temperature = None, # [K]
        conc = None, # [kmol/m^3]
        conc_ref = None, # [kmol/m^3]
    ):
        if isinstance(coeffs_Shomate, dict):
            coeffs_Shomate = list(coeffs_Shomate.values())
        self.coeffs_Shomate = coeffs_Shomate
        self.pressure_ref = pressure_ref
        super().__init__(
            temperature = temperature,
            conc = conc,
            conc_ref = conc_ref,
        )

    @property
    def enthalpy(self):
        temperature_red = self.temperature/1000
        return (
            +self.coeffs_Shomate[0] * temperature_red + 
            +self.coeffs_Shomate[1]/2 * temperature_red**2 + 
            +self.coeffs_Shomate[2]/3 * temperature_red**3 + 
            +self.coeffs_Shomate[3]/4 * temperature_red**4 + 
            -self.coeffs_Shomate[4] / temperature_red + 
            +self.coeffs_Shomate[5]
        ) * (units.kiloJoule/units.mole) # [J/kmol]

    @property
    def entropy_std(self):
        temperature_red = self.temperature/1000
        return (
            +self.coeffs_Shomate[0] * np.log(temperature_red) + 
            +self.coeffs_Shomate[1] * temperature_red + 
            +self.coeffs_Shomate[2]/2 * temperature_red**2 + 
            +self.coeffs_Shomate[3]/3 * temperature_red**3 + 
            -self.coeffs_Shomate[4] / (2*temperature_red**2) + 
            +self.coeffs_Shomate[6]
        ) * (units.Joule/units.mole/units.Kelvin) # [J/kmol/K]

    @property
    def specific_heat(self):
        temperature_red = self.temperature/1000
        return (
            +self.coeffs_Shomate[0] + 
            +self.coeffs_Shomate[1] * temperature_red + 
            +self.coeffs_Shomate[2] * temperature_red**2 + 
            +self.coeffs_Shomate[3] * temperature_red**3 + 
            +self.coeffs_Shomate[4] / temperature_red**2
        ) * (units.Joule/units.mole/units.Kelvin) # [J/kmol/K]

    def modify_energy(self, delta_energy):
        self.coeffs_Shomate[5] += delta_energy / (units.kiloJoule/units.mole)


# -----------------------------------------------------------------------------
# THERMO ASE
# -----------------------------------------------------------------------------


class ThermoAse(Thermo):

    def __init__(
        self,
        thermo_ase,
        pressure_ref = 1 * units.atm, # [Pa]
        temperature = None, # [K]
        conc = None, # [kmol/m^3]
        conc_ref = None, # [kmol/m^3]
    ):
        self.pressure_ref = pressure_ref
        self.thermo_ase = thermo_ase
        super().__init__(
            temperature = temperature,
            conc = conc,
            conc_ref = conc_ref,
        )

    @property
    def enthalpy(self):
        if hasattr(self.thermo_ase, 'get_enthalpy'):
            enthalpy = self.thermo_ase.get_enthalpy(
                temperature = self.temperature,
                verbose = False,
            )
        if hasattr(self.thermo_ase, 'get_internal_energy'):
            enthalpy = self.thermo_ase.get_internal_energy(
                temperature = self.temperature,
                verbose = False,
            )
        else:
            raise RuntimeError('thermo class cannot calculate enthalpy.')
        return enthalpy * units.eV/units.molecule # [J/kmol]
    
    @property
    def entropy_std(self):
        entropy = self.thermo_ase.get_entropy(
            temperature = self.temperature,
            verbose = False,
        )
        return entropy * units.eV/units.molecule # [J/kmol/K]

    def modify_energy(self, delta_energy):
        self.thermo_ase.potentialenergy += delta_energy / (units.eV/units.molecule)


# -----------------------------------------------------------------------------
# THERMO ND GAS
# -----------------------------------------------------------------------------


class ThermoNDgas(Thermo):

    def __init__(
        self,
        species,
        thermo_3Dgas = None,
        name_analyzer = None,
        pressure_ref = 1 * units.atm, # [Pa]
        temperature = None, # [K]
        conc = None, # [kmol/m^3]
        conc_ref = None, # [kmol/m^3]
    ):
        if isinstance(species, Species):
            self.thermo_3Dgas = species.thermo.copy()
        if thermo_3Dgas is not None:
            self.thermo_3Dgas = thermo_3Dgas.copy()
        molar_mass = get_molar_mass(species=species, name_analyzer=name_analyzer)
        self.atomic_mass = molar_mass/units.Navo # [kg]
        self.pressure_ref = pressure_ref
        super().__init__(
            temperature = temperature,
            conc = conc,
            conc_ref = conc_ref,
        )

    @Thermo.temperature.setter
    def temperature(self, temperature):
        self._temperature = temperature
        self.thermo_3Dgas.temperature = temperature

    @property
    def enthalpy(self):
        return self.thermo_3Dgas.enthalpy # [J/kmol]

    def modify_energy(self, delta_energy):
        self.thermo_3Dgas.modify_energy(delta_energy) # [J/kmol]


class Thermo2Dgas(ThermoNDgas):

    def __init__(
        self,
        species,
        conc_sites_surf, # [kmol/m^2]
        thermo_3Dgas = None,
        name_analyzer = None,
        pressure_ref = 1 * units.atm, # [Pa]
        temperature = None, # [K]
        conc = None, # [kmol/m^3]
        conc_ref = None, # [kmol/m^3]
    ):
        self.conc_sites_surf = conc_sites_surf
        super().__init__(
            species = species,
            thermo_3Dgas = thermo_3Dgas,
            name_analyzer = name_analyzer,
            pressure_ref = pressure_ref,
            temperature = temperature,
            conc = conc,
            conc_ref = conc_ref,
        )

    @property
    def entropy_std(self):
        return self.thermo_3Dgas.entropy_std + units.Rgas*np.log(
            self.pressure_ref/(units.Rgas*self.temperature)/self.conc_sites_surf
            * units.hP/(np.sqrt(2*np.pi*self.atomic_mass*units.kB*self.temperature))
        ) # [J/kmol/K]

    @property
    def specific_heat(self):
        # Calculated from d(entropy_std)/d(temperature).
        return self.thermo_3Dgas.specific_heat - 3/2*units.Rgas # [J/kmol/K]


class Thermo1Dgas(Thermo):

    def __init__(
        self,
        species,
        conc_sites_len, # [kmol/m]
        thermo_3Dgas = None,
        name_analyzer = None,
        pressure_ref = 1 * units.atm, # [Pa]
        temperature = None, # [K]
        conc = None, # [kmol/m^3]
        conc_ref = None, # [kmol/m^3]
    ):
        self.conc_sites_len = conc_sites_len
        super().__init__(
            species = species,
            thermo_3Dgas = thermo_3Dgas,
            name_analyzer = name_analyzer,
            pressure_ref = pressure_ref,
            temperature = temperature,
            conc = conc,
            conc_ref = conc_ref,
        )

    @property
    def entropy_std(self):
        return self.thermo_3Dgas.entropy_std + units.Rgas*np.log(
            self.pressure_ref/(units.Rgas*self.temperature)/self.conc_sites_len
            * units.hP**2/(2*np.pi*self.atomic_mass*units.kB*self.temperature)
        ) # [J/kmol/K]

    @property
    def specific_heat(self):
        # Calculated from d(entropy_std)/d(temperature).
        return self.thermo_3Dgas.specific_heat - 2*units.Rgas # [J/kmol/K]


class ThermoCombo(Thermo):

    def __init__(
        self,
        thermo_list,
        thermo_mult = None,
        temperature = None, # [K]
        conc = None, # [kmol/m^3]
        conc_ref = None, # [kmol/m^3]
    ):
        # thermo_list is a list of mikimoto.thermodynamics.Thermo classes
        # thermo_mult is a list of multiplication factors to calculate the 
        # thermodynamic properties of ThermoCombo from combinations of thermo_list.
        if thermo_mult is None:
            thermo_mult = [+1 for _ in len(thermo_list)]
        self.thermo_list = [thermo.copy() for thermo in thermo_list]
        self.thermo_mult = thermo_mult
        super().__init__(
            temperature = temperature,
            conc = conc,
            conc_ref = conc_ref,
        )

    @Thermo.temperature.setter
    def temperature(self, temperature):
        self._temperature = temperature
        for thermo in self.thermo_list:
            thermo.temperature = temperature

    @property
    def enthalpy(self):
        return np.sum([
            self.thermo_mult[ii]*thermo.enthalpy
            for ii, thermo in enumerate(self.thermo_list)
        ]) # [J/kmol]

    @property
    def entropy_std(self):
        return np.sum([
            self.thermo_mult[ii]*thermo.entropy_std
            for ii, thermo in enumerate(self.thermo_list)
        ]) # [J/kmol/K]

    @property
    def specific_heat(self):
        return np.sum([
            self.thermo_mult[ii]*thermo.specific_heat
            for ii, thermo in enumerate(self.thermo_list)
        ]) # [J/kmol/K]

    def modify_energy(self, delta_energy):
        self.thermo_list[0].modify_energy(delta_energy)


# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
