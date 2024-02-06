# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import io
import numpy as np
from contextlib import redirect_stdout
from scipy.integrate import LSODA, BDF, Radau, OdeSolution, RK45

# -----------------------------------------------------------------------------
# REACTORS
# -----------------------------------------------------------------------------

class IdealReactor:

    def __init__(
        self,
        microkin,
        update_kinetics = False,
        method = 'Radau',
        rtol = 1e-13,
        atol = 1e-15,
        t_bound = np.inf,
        conv_thr_ode = 1e-6,
        n_steps_max = np.inf,
        n_fails_max = 1e3,
        store_output = True,
    ):
        self.microkin = microkin
        self.update_kinetics = update_kinetics
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.t_bound = t_bound
        self.conv_thr_ode = conv_thr_ode
        self.n_steps_max = n_steps_max
        self.n_fails_max = n_fails_max
        self.store_output = store_output
    
    def initialize_solver(self, fun, t0, y0):
        
        method_dict = {
            'RK45': RK45,
            'Radau': Radau,
            'BDF': BDF,
            'LSODA': LSODA,
        }
        solver = method_dict[self.method](
            fun = fun,
            t0 = t0,
            y0 = y0,
            t_bound = self.t_bound,
            rtol = self.rtol,
            atol = self.atol,
        )
        
        return solver
    
    def integrate_ode(self, fun, t0, y0):
        """Integrate the system of ordinary differential equations."""
        n_steps = 0
        n_fails = 0
        solver = self.initialize_solver(fun, t0, y0)
        if self.store_output is True:
            self.initialize_results()
            self.store_results(time = t0)
        self.max_dy_dt = np.inf
        while (
            solver.status != 'finished' 
            and self.max_dy_dt > self.conv_thr_ode
            and n_steps < self.n_steps_max
        ):
            solver.step()
            if solver.status == 'failed':
                n_fails += 1
                y0 = solver.y + np.random.random(len(solver.y)) * 1e-6
                solver = self.initialize_solver(fun, solver.t, y0)
            else:
                n_steps += 1
                if self.store_output is True:
                    self.store_results(time = solver.t)
            if n_fails > self.n_fails_max:
                raise RuntimeError("Calculation Failed!")
    
        return solver.y

    def initialize_results(self):
        self.time_sol = []
        self.temperature_sol = []
        self.pressure_sol = []
        self.X_sol = []
        self.θ_sol = []
        self.conc_sol = []
        self.rho_sol = []

    def store_results(self, time):
        self.time_sol.append(time)
        self.temperature_sol.append(self.microkin.temperature)
        self.pressure_sol.append(self.microkin.pressure)
        self.X_sol.append(self.microkin.X_gas_list.copy())
        self.θ_sol.append(self.microkin.θ_surf_list.copy())
        self.conc_sol.append(self.microkin.conc_list.copy())
        self.rho_sol.append(self.microkin.rho_list.copy())

    def integrate_volume(self):
        pass


class CSTReactor(IdealReactor):
    
    def __init__(
        self,
        microkin,
        reactor_volume, # [m^3]
        vol_flow_rate, # [m^3/s]
        constant_temperature = True,
        constant_pressure = True,
        update_kinetics = False,
        method = 'Radau',
        rtol = 1e-13,
        atol = 1e-15,
        t_bound = np.inf,
        n_steps_max = np.inf,
        n_fails_max = 1e3,
        conv_thr_ode = 1e-6,
        store_output = True,
    ):
        self.microkin = microkin
        self.update_kinetics = update_kinetics
        self.constant_temperature = constant_temperature
        self.constant_pressure = constant_pressure
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.t_bound = t_bound # [s]
        self.conv_thr_ode = conv_thr_ode
        self.n_steps_max = n_steps_max
        self.n_fails_max = n_fails_max
        self.store_output = store_output
        self.vol_flow_rate_in = vol_flow_rate # [m^3/s]
        self.vol_flow_rate = vol_flow_rate # [m^3/s]
        self.reactor_volume_in = reactor_volume # [m^3]
        self.reactor_volume = reactor_volume # [m^3]
    
    def integrate_volume(self):
        
        # dmi_dt = mi_dot_in - mi_dot_out + sum(Rij)*volume
        # where mi are the masses, mi_dot_in and mi_dot_out
        # are the fluxes of mases entering and exiting the control volume
        # and are equal to the mass concentration entering and exiting
        # multiplied by the volumetric flow rate. Dividing by volume we get:
        # dρi_dt = (ρi_in*V_dot_in - ρi_out*V_dot_out)/volume + sum(Rij)*MWi
        # where V_dot_in and V_dot_out are the volumetric flow rates and MWi
        # is the molar mass.
        
        rho_in = np.array(self.microkin.rho_list.copy()) # [kg/m^3]
        rho_zero = rho_in.copy() # [kg/m^3]
        mass_flow_rate = self.vol_flow_rate_in*self.microkin.rho_gas_tot # [kg/s]
        moles_gas_zero = self.reactor_volume_in*self.microkin.conc_gas_tot # [kmol]
        
        if self.constant_temperature is False:
            raise NotImplementedError("Adiabatic reactors not implemented.")

        def fun_drho_dt(time, rho_list):
            if self.update_kinetics is True:
                self.microkin.get_kinetic_constants()
            self.microkin.rho_list = rho_list # [kg/m^3]
            if self.constant_pressure is True:
                self.reactor_volume = moles_gas_zero/self.microkin.conc_gas_tot # [m^3]
                self.microkin.conc_list = np.array(self.microkin.conc_list) * (
                    self.reactor_volume/self.reactor_volume_in
                ) # [kmol/m^3]
            self.microkin.get_reaction_rates() # [kmol/m^3/s]
            prod_rates = self.microkin.get_net_production_rates() # [kmol/m^3/s]
            self.vol_flow_rate = mass_flow_rate/self.microkin.rho_gas_tot # [m^3/s]
            drho_dt = prod_rates*self.microkin.molar_mass_list # [kg/m^3/s]
            drho_dt += self.microkin.is_gas * (
                +self.vol_flow_rate_in*rho_in/self.reactor_volume
                -self.vol_flow_rate*rho_list/self.reactor_volume
            ) # [kg/m^3/s]
            self.max_dy_dt = np.max(drho_dt) # [kg/m^3/s]
            return drho_dt
        
        # Solve the system of differential equations.
        self.microkin.get_kinetic_constants()
        rho_out = self.integrate_ode(
            fun = fun_drho_dt,
            t0 = 0.,
            y0 = rho_zero.copy(),
        )
        

class BatchReactor(CSTReactor):

    def __init__(
        self,
        microkin,
        reactor_volume,
        constant_temperature = True,
        constant_pressure = False,
        update_kinetics = False,
        method = 'Radau',
        rtol = 1e-13,
        atol = 1e-15,
        t_bound = np.inf,
        n_steps_max = np.inf,
        n_fails_max = 1e3,
        conv_thr_ode = 1e-6,
        store_output = True,
    ):
        super().__init__(
            microkin = microkin,
            reactor_volume = reactor_volume,
            constant_temperature = constant_temperature,
            constant_pressure = constant_pressure,
            vol_flow_rate = 0.,
            update_kinetics = update_kinetics,
            method = method,
            rtol = rtol,
            atol = atol,
            t_bound = t_bound,
            n_steps_max = n_steps_max,
            n_fails_max = n_fails_max,
            conv_thr_ode = conv_thr_ode,
        )


class PFReactor(IdealReactor):
    def __init__(
        self,
        microkin,
        reactor_volume,
        vol_flow_rate,
        constant_temperature = True,
        constant_pressure = True,
        delta_time = 1e-7,
        update_kinetics = False,
        method = 'RK45',
        rtol = 1e-08,
        atol = 1e-10,
        conv_thr_ode = 0.,
        n_steps_max = np.inf,
        n_fails_max = 1e3,
        method_cstr = 'Radau',
        rtol_cstr = 1e-13,
        atol_cstr = 1e-15,
        conv_thr_ode_cstr = 1e-6,
        store_output = True,
    ):
        self.microkin = microkin
        self.reactor_volume = reactor_volume # [m^3]
        self.vol_flow_rate = vol_flow_rate # [m^3/s]
        self.constant_temperature = constant_temperature
        self.constant_pressure = constant_pressure
        self.delta_time = delta_time # [s]
        self.update_kinetics = update_kinetics
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.conv_thr_ode = conv_thr_ode
        self.n_steps_max = n_steps_max
        self.n_fails_max = n_fails_max
        self.method_cstr = method_cstr
        self.rtol_cstr = rtol_cstr
        self.atol_cstr = atol_cstr
        self.conv_thr_ode_cstr = conv_thr_ode_cstr
        self.store_output = store_output
        self.cstr_volume = vol_flow_rate*delta_time # [m^3]
        self.t_bound = reactor_volume/vol_flow_rate # [s]

    def integrate_volume(self):

        cstr = CSTReactor(
            microkin = self.microkin,
            reactor_volume = self.cstr_volume,
            vol_flow_rate = self.vol_flow_rate,
            constant_temperature = self.constant_temperature,
            constant_pressure = self.constant_pressure,
            t_bound = np.inf,
            method = self.method_cstr,
            rtol = self.rtol_cstr,
            atol = self.atol_cstr,
            conv_thr_ode = self.conv_thr_ode_cstr,
            store_output = False,
        )

        X_in = np.array(self.microkin.X_gas_list)
        θ_in = np.array(self.microkin.θ_surf_list)
        
        def fun_dconc_dt(time, conc_gas):
            X_in = conc_gas/self.microkin.conc_gas_tot
            self.microkin.X_gas_list = X_in.copy()
            cstr.integrate_volume()
            X_out = self.microkin.X_gas_list
            self.delta_time = cstr.reactor_volume/cstr.vol_flow_rate
            dconc_dt = (X_out-X_in)*self.microkin.conc_gas_tot/self.delta_time
            return dconc_dt
        
        # Solve the system of differential equations.
        self.microkin.get_kinetic_constants()
        conc_out = self.integrate_ode(
            fun = fun_dconc_dt,
            t0 = 0.,
            y0 = X_in*self.microkin.conc_gas_tot,
        )
        

class PFReactorSeriesCSTR(IdealReactor):
    def __init__(
        self,
        microkin,
        n_cstr,
        reactor_volume,
        reactor_length,
        vol_flow_rate,
        constant_temperature = True,
        constant_pressure = True,
        method_cstr = 'Radau',
        rtol_cstr = 1e-13,
        atol_cstr = 1e-15,
        conv_thr_ode_cstr = 1e-6,
        store_output = True,
        print_output = True,
        print_coverages = False,
        n_print = 10,
    ):
        self.microkin = microkin
        self.n_cstr = n_cstr
        self.reactor_volume = reactor_volume # [m^3]
        self.reactor_length = reactor_length # [m]
        self.vol_flow_rate = vol_flow_rate # [m^3/s]
        self.constant_temperature = constant_temperature
        self.constant_pressure = constant_pressure
        self.cstr_length = reactor_length/n_cstr # [m]
        self.cstr_volume = reactor_volume/n_cstr # [m^3]
        self.method_cstr = method_cstr
        self.rtol_cstr = rtol_cstr
        self.atol_cstr = atol_cstr
        self.conv_thr_ode_cstr = conv_thr_ode_cstr
        self.store_output = store_output
        self.print_output = print_output
        self.print_coverages = print_coverages
        self.n_print = n_print

    def integrate_volume(self):

        cstr = CSTReactor(
            microkin = self.microkin,
            reactor_volume = self.cstr_volume,
            vol_flow_rate = self.vol_flow_rate,
            constant_temperature = self.constant_temperature,
            constant_pressure = self.constant_pressure,
            t_bound = np.inf,
            method = self.method_cstr,
            rtol = self.rtol_cstr,
            atol = self.atol_cstr,
            conv_thr_ode = self.conv_thr_ode_cstr,
            store_output = False,
        )

        X_in = np.array(self.microkin.X_gas_list)
        θ_in = np.array(self.microkin.θ_surf_list)
        X_zero = X_in.copy()
        θ_zero = θ_in.copy()
        
        if self.store_output is True:
            self.initialize_results()
            self.store_results(time = 0.)
        
        if self.print_output is True:
            self.print_X_and_θ(z_reactor = 0., print_names = True)
        
        for ii in range(self.n_cstr):
            
            # Integrate the control volume.
            z_reactor = (ii+1)*self.cstr_length
            cstr.integrate_volume()
            
            if self.store_output is True:
                time = (ii+1)*self.cstr_volume/self.vol_flow_rate
                self.store_results(time = time)
            
            # Print to screen the gas composition.
            if self.print_output is True:
                print_int = int(self.n_cstr/self.n_print) if self.n_cstr > 1 else 1
                if print_int > 0 and (ii+1) % print_int == 0:
                    self.print_X_and_θ(z_reactor = z_reactor, print_names = False)

    def print_X_and_θ(self, z_reactor, print_names = False):
        if print_names is True:
            string = 'distance[m]'.rjust(14)
            for spec in self.microkin.gas_species:
                string += ('X_'+spec.name).rjust(12)
            if self.print_coverages is True:
                for spec in self.microkin.surf_species:
                    string += ('θ_'+spec.name).rjust(12)
            print(string)
        string = f'  {z_reactor:12f}'
        for ii, spec in enumerate(self.microkin.gas_species):
            string += f'  {self.microkin.X_gas_list[ii]:10.6f}'
        if self.print_coverages is True:
            for ii, spec in enumerate(self.microkin.surf_species):
                string += f'  {self.microkin.θ_surf_list[ii]:10.6f}'
        print(string)

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
