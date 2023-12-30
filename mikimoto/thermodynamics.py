# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import cantera as ct
import numpy as np
import copy as cp
from mikimoto import units
from mikimoto.microkinetics import Species
from mikimoto.utilities import get_molecular_mass

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
        self.temperature = temperature
        self.conc = conc
        self.conc_ref = conc_ref

    @property
    def temperature(self):
        if self._temperature is None:
            raise RuntimeError("temperature not set.")
        return self._temperature

    @temperature.setter
    def temperature(self, temperature):
        self._temperature = temperature

    @property
    def conc(self):
        if self._conc is None:
            raise RuntimeError("conc not set.")
        return self._conc

    @conc.setter
    def conc(self, conc):
        self._conc = conc

    @property
    def conc_ref(self):
        if self._conc_ref is None:
            raise RuntimeError("conc_ref not set.")
        return self._conc_ref

    @conc_ref.setter
    def conc_ref(self, conc_ref):
        self._conc_ref = conc_ref

    @property
    def enthalpy(self):
        pass

    @property
    def entropy_std(self):
        pass

    @property
    def specific_heat(self):
        pass

    @property
    def Gibbs_std(self):
        return self.enthalpy - self.temperature * self.entropy_std

    @property
    def entropy(self):
        molfract = self.conc/self.conc_ref
        log_molfract = np.log(molfract) if molfract > 0. else -np.inf
        return self.entropy_std + units.Rgas * self.temperature * log_molfract

    @property
    def Gibbs(self):
        return self.enthalpy - self.temperature * self.entropy

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
        self.Gibbs_ref = Gibbs_ref
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
        return self.Gibbs_ref

    @property
    def entropy(self):
        raise RuntimeError("No entropy available in ThermoFixedG0.")

    @property
    def Gibbs(self):
        molfract = self.conc/self.conc_ref
        log_molfract = np.log(molfract) if molfract > 0. else -np.inf
        return self.Gibbs_std + units.Rgas * self.temperature * log_molfract

    def modify_energy(self, delta_energy):
        self.Gibbs_ref += delta_energy


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
        self.enthalpy_ref = enthalpy_ref
        self.entropy_ref = entropy_ref
        self.specific_heat_ref = specific_heat_ref
        self.temperature_ref = temperature_ref
        self.pressure_ref = pressure_ref
        super().__init__(
            temperature = temperature,
            conc = conc,
            conc_ref = conc_ref,
        )

    @property
    def enthalpy(self):
        return self.enthalpy_ref + self.specific_heat_ref * (
            self.temperature - self.temperature_ref
        )

    @property
    def entropy_std(self):
        return self.entropy_ref + self.specific_heat_ref * np.log(
            self.temperature / self.temperature_ref
        )

    @property
    def specific_heat(self):
        return self.specific_heat_ref

    def modify_energy(self, delta_energy):
        self.enthalpy_ref += delta_energy


class ThermoNASA7(Thermo):

    def __init__(
        self,
        coeffs_NASA,
        pressure_ref = 1 * units.atm, # [Pa]
        temperature = None, # [K]
        conc = None, # [kmol/m^3]
        conc_ref = None, # [kmol/m^3]
    ):
        self.coeffs_NASA = coeffs_NASA
        self.pressure_ref = pressure_ref
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
        )

    @property
    def entropy_std(self):
        return units.Rgas * (
            self.coeffs_NASA[0] * np.log(self.temperature) + 
            self.coeffs_NASA[1] * self.temperature + 
            self.coeffs_NASA[2]/2 * self.temperature**2 + 
            self.coeffs_NASA[3]/3 * self.temperature**3 + 
            self.coeffs_NASA[4]/4 * self.temperature**4 + 
            self.coeffs_NASA[6]
        )

    @property
    def specific_heat(self):
        return units.Rgas * (
            self.coeffs_NASA[0] + 
            self.coeffs_NASA[1] * self.temperature + 
            self.coeffs_NASA[2] * self.temperature**2 + 
            self.coeffs_NASA[3] * self.temperature**3 + 
            self.coeffs_NASA[4] * self.temperature**4
        )

    def modify_energy(self, delta_energy):
        self.coeffs_NASA[5] += delta_energy/units.Rgas


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
        return enthalpy * units.eV/units.molecule
    
    @property
    def entropy_std(self):
        entropy = self.thermo_ase.get_entropy(
            temperature = self.temperature,
            verbose = False,
        )
        return entropy * units.eV/units.molecule

    def modify_energy(self, delta_energy):
        self.thermo_ase.potentialenergy += delta_energy / (units.eV/units.molecule)


# -----------------------------------------------------------------------------
# THERMO CANTERA
# -----------------------------------------------------------------------------


class ThermoCantera(Thermo):
    
    def __init__(
        self,
        temperature = None, # [K]
        conc = None, # [kmol/m^3]
        conc_ref = None, # [kmol/m^3]
    ):
        super().__init__(
            temperature = temperature,
            conc = conc,
            conc_ref = conc_ref,
        )

    @property
    def enthalpy(self):
        return self.thermo_ct.h(self.temperature)

    @property
    def entropy_std(self):
        return self.thermo_ct.s(self.temperature)

    @property
    def specific_heat(self):
        return self.thermo_ct.cp(self.temperature)

    def update_thermo_ct(self):
        pass

    def copy(self):
        # This is because cp.deepcopy does not work with Cantera Cython classes.
        del self.thermo_ct
        copy = cp.deepcopy(self)
        self.update_thermo_ct()
        copy.update_thermo_ct()
        return copy


class ThermoCanteraConstantCp(ThermoCantera):

    def __init__(
        self,
        enthalpy_ref, # [J/kmol]
        entropy_ref, # [J/kmol/K]
        specific_heat_ref, # [J/kmol/K]
        temperature_ref, # [K]
        temperature_low = 200.00, # [K]
        temperature_high = 5000.00, # [K]
        pressure_ref = ct.one_atm, # [Pa]
        temperature = None, # [K]
        conc = None, # [kmol/m^3]
        conc_ref = None, # [kmol/m^3]
    ):
        self.temperature_ref = temperature_ref
        self.pressure_ref = pressure_ref
        self.temperature_low = temperature_low
        self.temperature_high = temperature_high
        self.coeffs = [temperature_ref, enthalpy_ref, entropy_ref, specific_heat_ref]
        self.update_thermo_ct()
        super().__init__(
            temperature = temperature,
            conc = conc,
            conc_ref = conc_ref,
        )

    def update_thermo_ct(self):
        self.thermo_ct = ct.ConstantCp(
            T_low = self.temperature_low,
            T_high = self.temperature_high,
            P_ref = self.pressure_ref,
            coeffs = self.coeffs,
        )

    def modify_energy(self, delta_energy):
        self.coeffs[1] += delta_energy
        self.update_thermo_ct()


class ThermoCanteraNASA7(ThermoCantera):

    def __init__(
        self,
        coeffs_NASA,
        temperature_low = 200.00, # [K]
        temperature_mid = 2000.00, # [K]
        temperature_high = 5000.00, # [K]
        pressure_ref = ct.one_atm, # [Pa]
        temperature = None, # [K]
        conc = None, # [kmol/m^3]
        conc_ref = None, # [kmol/m^3]
    ):
        self.pressure_ref = pressure_ref
        self.temperature_low = temperature_low
        self.temperature_high = temperature_high
        if len(coeffs_NASA) == 7:
            self.coeffs = [temperature_mid]+list(coeffs_NASA)*2
        elif len(coeffs_NASA) == 14:
            self.coeffs = [temperature_mid]+list(coeffs_NASA)
        else:
            self.coeffs = list(coeffs_NASA)
        self.update_thermo_ct()
        super().__init__(
            temperature = temperature,
            conc = conc,
            conc_ref = conc_ref,
        )

    def update_thermo_ct(self):
        self.thermo_ct = ct.NasaPoly2(
            T_low = self.temperature_low,
            T_high = self.temperature_high,
            P_ref = self.pressure_ref,
            coeffs = self.coeffs,
        )

    def modify_energy(self, delta_energy):
        self.coeffs[13] += delta_energy/ct.gas_constant
        self.coeffs[6] += delta_energy/ct.gas_constant
        self.update_thermo_ct()
    

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
        molec_mass = get_molecular_mass(species=species, name_analyzer=name_analyzer)
        self.atomic_mass = molec_mass/units.Navo # [kg]
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
        return self.thermo_3Dgas.enthalpy

    def modify_energy(self, delta_energy):
        self.thermo_3Dgas.modify_energy(delta_energy)


class Thermo2Dgas(ThermoNDgas):

    def __init__(
        self,
        species,
        sites_surf_conc, # [kmol/m^2]
        thermo_3Dgas = None,
        name_analyzer = None,
        pressure_ref = 1 * units.atm, # [Pa]
        temperature = None, # [K]
        conc = None, # [kmol/m^3]
        conc_ref = None, # [kmol/m^3]
    ):
        self.sites_surf_conc = sites_surf_conc
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
            self.pressure_ref/(units.Rgas*self.temperature)/self.sites_surf_conc
            * units.hP/(np.sqrt(2*np.pi*self.atomic_mass*units.kB*self.temperature))
        )

    @property
    def specific_heat(self):
        # Calculated from d(entropy_std)/d(temperature).
        return self.thermo_3Dgas.specific_heat - 3/2*units.Rgas


class Thermo1Dgas(Thermo):

    def __init__(
        self,
        species,
        sites_len_conc, # [kmol/m]
        thermo_3Dgas = None,
        name_analyzer = None,
        pressure_ref = 1 * units.atm, # [Pa]
        temperature = None, # [K]
        conc = None, # [kmol/m^3]
        conc_ref = None, # [kmol/m^3]
    ):
        self.sites_len_conc = sites_len_conc
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
            self.pressure_ref/(units.Rgas*self.temperature)/self.sites_len_conc
            * units.hP**2/(2*np.pi*self.atomic_mass*units.kB*self.temperature)
        )

    @property
    def specific_heat(self):
        # Calculated from d(entropy_std)/d(temperature).
        return self.thermo_3Dgas.specific_heat - 2*units.Rgas


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
        ])

    @property
    def entropy_std(self):
        return np.sum([
            self.thermo_mult[ii]*thermo.entropy_std
            for ii, thermo in enumerate(self.thermo_list)
        ])

    @property
    def specific_heat(self):
        return np.sum([
            self.thermo_mult[ii]*thermo.specific_heat
            for ii, thermo in enumerate(self.thermo_list)
        ])

    def modify_energy(self, delta_energy):
        self.thermo_list[0].modify_energy(delta_energy)


# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
