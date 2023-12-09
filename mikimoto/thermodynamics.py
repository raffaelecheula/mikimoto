# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import cantera as ct
import numpy as np
from mikimoto import units

# -----------------------------------------------------------------------------
# THERMO
# -----------------------------------------------------------------------------


class Thermo:

    def __init__(
        self,
        temperature = None,
        conc = None,
        conc_ref = None,
    ):
        self._temperature = temperature
        self._conc = conc
        self._conc_ref = conc_ref

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
        return self.enthalpy-self.temperature*self.entropy_std

    @property
    def entropy(self):
        molfract = self.conc/self.conc_ref
        log_molfract = np.log(molfract) if molfract > 0. else -np.inf
        return self.entropy_std + units.Rgas * self.temperature * log_molfract

    @property
    def Gibbs(self):
        return self.enthalpy-self.temperature*self.entropy

    def modify_energy(self, delta_energy):
        pass


class ThermoFixedG0(Thermo):

    def __init__(
        self,
        Gibbs_ref,
        pressure_ref = 1 * units.atm, # [Pa]
        temperature = None,
        conc = None,
        conc_ref = None,
    ):
        super().__init__(
            temperature = temperature,
            conc = conc,
            conc_ref = conc_ref,
        )
        self.Gibbs_ref = Gibbs_ref
        self.pressure_ref = pressure_ref

    @property
    def enthalpy(self):
        raise RuntimeError("No enthalpy available in ThermoFixedG0.")

    @property
    def entropy_std(self):
        raise RuntimeError("No entropy_std available in ThermoFixedG0.")

    @property
    def specific_heat(self, temperature):
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
        enthalpy_ref,
        entropy_ref,
        specific_heat_ref,
        temperature_ref,
        pressure_ref = 1 * units.atm, # [Pa]
        temperature = None,
        conc = None,
        conc_ref = None,
    ):
        super().__init__(
            temperature = temperature,
            conc = conc,
            conc_ref = conc_ref,
        )
        self.enthalpy_ref = enthalpy_ref
        self.entropy_ref = entropy_ref
        self.specific_heat_ref = specific_heat_ref
        self.temperature_ref = temperature_ref
        self.pressure_ref = pressure_ref

    @property
    def enthalpy(self):
        return self.enthalpy_ref + self.specific_heat_ref * (
            self.temperature - self.temperature_ref
        )

    @property
    def entropy_std(self, temperature):
        return self.entropy_ref + self.specific_heat_ref * np.log(
            temperature / self.temperature_ref
        )

    @property
    def get_specific_heat(self):
        return self.specific_heat_ref

    def modify_energy(self, delta_energy):
        self.enthalpy_ref += delta_energy


class ThermoNASA7(Thermo):

    def __init__(
        self,
        coeffs_NASA,
        pressure_ref = 1 * units.atm, # [Pa]
        temperature = None,
        conc = None,
        conc_ref = None,
    ):
        super().__init__(
            temperature = temperature,
            conc = conc,
            conc_ref = conc_ref,
        )
        self.coeffs_NASA = coeffs_NASA
        self.pressure_ref = pressure_ref

    @property
    def enthalpy(self, temperature):
        return units.Rgas * (
            self.coeffs_NASA[0] * temperature + 
            self.coeffs_NASA[1]/2 * temperature**2 + 
            self.coeffs_NASA[2]/3 * temperature**3 + 
            self.coeffs_NASA[3]/4 * temperature**4 + 
            self.coeffs_NASA[4]/5 * temperature**5 + 
            self.coeffs_NASA[5]
        )

    @property
    def entropy_std(self, temperature):
        return units.Rgas * (
            self.coeffs_NASA[0] * np.log(temperature) + 
            self.coeffs_NASA[1] * temperature + 
            self.coeffs_NASA[2]/2 * temperature**2 + 
            self.coeffs_NASA[3]/3 * temperature**3 + 
            self.coeffs_NASA[4]/4 * temperature**4 + 
            self.coeffs_NASA[6]
        )

    @property
    def get_specific_heat(self, temperature):
        return units.Rgas * (
            self.coeffs_NASA[0] + 
            self.coeffs_NASA[1] * temperature + 
            self.coeffs_NASA[2] * temperature**2 + 
            self.coeffs_NASA[3] * temperature**3 + 
            self.coeffs_NASA[4] * temperature**4
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
        temperature = None,
        conc = None,
        conc_ref = None,
    ):
        super().__init__(
            temperature = temperature,
            conc = conc,
            conc_ref = conc_ref,
        )
        self.pressure_ref = pressure_ref
        self.thermo_ase = thermo_ase

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
        temperature = None,
        conc = None,
        conc_ref = None,
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
        return self.thermo_ct.cP(self.temperature)


class ThermoCanteraConstantCp(ThermoCantera):

    def __init__(
        self,
        enthalpy_ref,
        entropy_ref,
        specific_heat_ref,
        temperature_ref,
        temperature_low = 200.00,
        temperature_high = 5000.00,
        pressure_ref = ct.one_atm, # [Pa]
        temperature = None,
        conc = None,
        conc_ref = None,
    ):
        
        super().__init__(
            temperature = temperature,
            conc = conc,
            conc_ref = conc_ref,
        )
        self.temperature_ref = temperature_ref
        self.pressure_ref = pressure_ref
        self.temperature_low = temperature_low
        self.temperature_high = temperature_high
        self.coeffs = [temperature_ref, enthalpy_ref, entropy_ref, specific_heat_ref]
        self.thermo_ct = ct.ConstantCp(
            T_low = self.temperature_low,
            T_high = self.temperature_high,
            P_ref = self.pressure_ref,
            coeffs = self.coeffs,
        )

    def modify_energy(self, delta_energy):
        self.coeffs[1] += delta_energy
        self.thermo_ct = ct.ConstantCp(
            T_low = self.temperature_low,
            T_high = self.temperature_high,
            P_ref = self.pressure_ref,
            coeffs = self.coeffs,
        )


class ThermoCanteraNASA7(ThermoCantera):

    def __init__(
        self,
        coeffs_NASA,
        temperature_low = 200.00,
        temperature_mid = 2000.00,
        temperature_high = 5000.00,
        pressure_ref = ct.one_atm, # [Pa]
        temperature = None,
        conc = None,
        conc_ref = None,
    ):
        super().__init__(
            temperature = temperature,
            conc = conc,
            conc_ref = conc_ref,
        )
        self.pressure_ref = pressure_ref
        self.temperature_low = temperature_low
        self.temperature_high = temperature_high
        if len(coeffs_NASA) == 7:
            self.coeffs = [temperature_mid]+list(coeffs_NASA)*2
        elif len(coeffs_NASA) == 14:
            self.coeffs = [temperature_mid]+list(coeffs_NASA)
        else:
            self.coeffs = list(coeffs_NASA)
        self.thermo_ct = ct.NasaPoly2(
            T_low = self.temperature_low,
            T_high = self.temperature_high,
            P_ref = self.pressure_ref,
            coeffs = self.coeffs,
        )

    def modify_energy(self, delta_energy):
        self.coeffs[13] += delta_energy/ct.gas_constant
        self.coeffs[6] += delta_energy/ct.gas_constant
        self.thermo_ct = ct.NasaPoly2(
            T_low = self.temperature_low,
            T_high = self.temperature_high,
            P_ref = self.pressure_ref,
            coeffs = self.coeffs,
        )


# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
