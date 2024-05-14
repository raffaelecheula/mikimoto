# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import copy as cp
import cantera as ct
from mikimoto import units
from mikimoto.thermodynamics import Thermo

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
        return self.thermo_ct.h(self.temperature) # [J/kmol]

    @property
    def entropy_std(self):
        return self.thermo_ct.s(self.temperature) # [J/kmol/K]

    @property
    def specific_heat(self):
        return self.thermo_ct.cp(self.temperature) # [J/kmol/K]

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
        self.coeffs[13] += delta_energy / ct.gas_constant
        self.coeffs[6] += delta_energy / ct.gas_constant
        self.update_thermo_ct()
    

class ThermoCanteraShomate(ThermoCantera):

    def __init__(
        self,
        coeffs_Shomate,
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
        if len(coeffs_Shomate) == 7:
            self.coeffs = [temperature_mid]+list(coeffs_Shomate)*2
        elif len(coeffs_Shomate) == 14:
            self.coeffs = [temperature_mid]+list(coeffs_Shomate)
        else:
            self.coeffs = list(coeffs_Shomate)
        self.update_thermo_ct()
        super().__init__(
            temperature = temperature,
            conc = conc,
            conc_ref = conc_ref,
        )

    def update_thermo_ct(self):
        self.thermo_ct = ct.ShomatePoly2(
            T_low = self.temperature_low,
            T_high = self.temperature_high,
            P_ref = self.pressure_ref,
            coeffs = self.coeffs,
        )

    def modify_energy(self, delta_energy):
        self.coeffs[13] += delta_energy / (units.kiloJoule/units.mole)
        self.coeffs[6] += delta_energy / (units.kiloJoule/units.mole)
        self.update_thermo_ct()


# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
