# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
from scipy.integrate import LSODA, BDF, Radau, OdeSolution, RK45

# -----------------------------------------------------------------------------
# REACTORS
# -----------------------------------------------------------------------------

class IdealReactor:

    def __init__(
        self,
        microkin,
        update_kinetics = False,
        method = 'LSODA',
        rtol = 1e-08,
        atol = 1e-10,
        t_bound = np.inf,
        conv_thr_ode = 1e-9,
        n_steps_max = np.inf,
        n_fails_max = 1e3,
        store_output = True,
        dense_output = False,
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
        self.dense_output = dense_output
    
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
        
        n_steps = 0
        n_fails = 0
        solver = self.initialize_solver(fun, t0, y0)
        if self.store_output is True:
            self.store_results(solver, n_steps)
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
                solver = self.initialize_solver(fun, solver.t, y0) # solver.t_old?
            else:
                n_steps += 1
                if self.store_output is True:
                    self.store_results(solver, n_steps)
            if n_fails > self.n_fails_max:
                raise RuntimeError("Calculation Failed!")
        if self.dense_output is True:
            self.conc_out_fun = OdeSolution(self.time_vect, self.interp_vect)
    
        return solver.y

    def store_results(self, solver, n_steps):
        if n_steps == 0:
            self.time_vect = [solver.t]
            self.conc_vect = [solver.y] # TODO:
            self.temperature_vect = [self.microkin.temperature]
            self.pressure_vect = [self.microkin.pressure]
            self.X_vect = [self.microkin.X_list.copy()]
            self.θ_vect = [self.microkin.θ_list.copy()]
            if self.dense_output is True:
                self.interp_vect = []
        else:
            self.time_vect.append(solver.t)
            self.conc_vect.append(solver.y)
            self.temperature_vect.append(self.microkin.temperature)
            self.pressure_vect.append(self.microkin.pressure)
            self.X_vect.append(self.microkin.X_list.copy())
            self.θ_vect.append(self.microkin.θ_list.copy())
            if self.dense_output is True:
                self.interp_vect.append(solver.dense_output())

    def integrate_volume(self):
        pass


class CSTReactor(IdealReactor):
    
    def __init__(
        self,
        microkin,
        reactor_volume, # [m^3]
        vol_flow_rate, # [m^3/s]
        update_kinetics = False,
        method = 'LSODA',
        rtol = 1e-10,
        atol = 1e-12,
        t_bound = np.inf,
        n_steps_max = np.inf,
        n_fails_max = 1e3,
        conv_thr_ode = 1e-8,
        store_output = True,
        dense_output = False,
    ):
        self.microkin = microkin
        self.update_kinetics = update_kinetics
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.t_bound = t_bound # [s]
        self.conv_thr_ode = conv_thr_ode
        self.n_steps_max = n_steps_max
        self.n_fails_max = n_fails_max
        self.store_output = store_output
        self.dense_output = dense_output
        self.vol_flow_rate = vol_flow_rate # [m^3/s]
        self.vol_flow_rate_out = vol_flow_rate # [m^3/s]
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
        
        rho_in = np.array(self.microkin.rho_list.copy())
        rho_zero = rho_in.copy()
        mass_flow_rate = self.vol_flow_rate*self.microkin.rho_gas_tot
        
        def fun_drho_dt(time, rho_list):
            if self.update_kinetics is True:
                self.microkin.get_kinetic_constants()
            self.microkin.rho_list = rho_list # [kg/m^3]
            self.microkin.get_reaction_rates() # [kmol/m^3/s]
            prod_rates = self.microkin.get_net_production_rates() # [kmol/m^3/s]
            self.vol_flow_rate_out = mass_flow_rate/self.microkin.rho_gas_tot
            drho_dt = prod_rates*self.microkin.molar_mass_list
            drho_dt += self.microkin.is_gas * (
                +self.vol_flow_rate*rho_in/self.reactor_volume
                -self.vol_flow_rate_out*rho_list/self.reactor_volume
            )
            self.max_dy_dt = np.max(drho_dt)
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
        update_kinetics = False,
        method = 'LSODA',
        rtol = 1e-10,
        atol = 1e-12,
        t_bound = np.inf,
        n_steps_max = np.inf,
        n_fails_max = 1e3,
        conv_thr_ode = 1e-8,
        store_output = True,
        dense_output = False,
    ):
        super().__init__(
            microkin = microkin,
            reactor_volume = reactor_volume,
            vol_flow_rate = 0.,
            update_kinetics = update_kinetics,
            method = method,
            rtol = rtol,
            atol = atol,
            t_bound = t_bound,
            n_steps_max = n_steps_max,
            n_fails_max = n_fails_max,
            conv_thr_ode = conv_thr_ode,
            dense_output = dense_output,
        )


class PFReactor(IdealReactor):
    def __init__(
        self,
        microkin,
        reactor_volume,
        vol_flow_rate,
        delta_time = 1e-7,
        update_kinetics = False,
        method = 'RK45',
        rtol = 1e-08,
        atol = 1e-10,
        conv_thr_ode = 0.,
        n_steps_max = np.inf,
        n_fails_max = 1e3,
        method_cstr = 'LSODA',
        rtol_cstr = 1e-10,
        atol_cstr = 1e-12,
        store_output = True,
        dense_output = False,
    ):
        self.microkin = microkin
        self.reactor_volume = reactor_volume # [m^3]
        self.vol_flow_rate = vol_flow_rate # [m^3/s]
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
        self.store_output = store_output
        self.dense_output = dense_output
        self.cstr_volume = vol_flow_rate*delta_time # [m^3]
        self.t_bound = reactor_volume/vol_flow_rate # [s]

    def integrate_volume(
        self,
        X_in = None, # [-]
        θ_in = None, # [-]
    ):

        X_in = np.array(self.microkin.X_list) if X_in is None else np.array(X_in)
        θ_in = np.array(self.microkin.θ_list) if θ_in is None else np.array(θ_in)
        
        cstr = CSTReactor(
            microkin = self.microkin,
            reactor_volume = self.cstr_volume,
            vol_flow_rate = self.vol_flow_rate,
            t_bound = np.inf,
            method = self.method_cstr,
            rtol = self.rtol_cstr,
            atol = self.atol_cstr,
            store_output = False,
            dense_output = False,
        )

        self.X_zero = X_in.copy()
        self.θ_zero = θ_in.copy()
        def fun_dconc_dt(time, conc_gas):
            X_in = conc_gas/self.microkin.conc_tot_gas
            X_out, θ_out = cstr.integrate_volume(
                X_in = X_in,
                θ_in = θ_in,
                X_zero = self.X_zero,
                θ_zero = self.θ_zero,
            )
            self.X_zero = X_out.copy()
            self.θ_zero = θ_out.copy()
            dconc_dt = (X_out-X_in)*self.microkin.conc_tot_gas/self.delta_time
            return dconc_dt
        
        # Solve the system of differential equations.
        self.microkin.get_kinetic_constants()
        conc_out = self.integrate_ode(
            fun = fun_dconc_dt,
            t0 = 0.,
            y0 = X_in*self.microkin.conc_tot_gas,
        )
        
        return self.X_zero, self.θ_zero


class PFReactorSeriesCSTR:
    def __init__(
        self,
        microkin,
        n_cstr,
        reactor_volume,
        reactor_length,
        vol_flow_rate,
        method_cstr = 'LSODA',
        rtol_cstr = 1e-10,
        atol_cstr = 1e-12,
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
        self.cstr_length = reactor_length/n_cstr # [m]
        self.cstr_volume = reactor_volume/n_cstr # [m^3]
        self.method_cstr = method_cstr
        self.rtol_cstr = rtol_cstr
        self.atol_cstr = atol_cstr
        self.store_output = store_output
        self.print_output = print_output
        self.print_coverages = print_coverages
        self.n_print = n_print

    def print_X_and_θ(
        self,
        z_reactor,
        X_array,
        θ_array,
        print_names = False,
    ):
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
            string += f'  {X_array[ii]:10.6f}'
        if self.print_coverages is True:
            for ii, spec in enumerate(self.microkin.surf_species):
                string += f'  {θ_array[ii]:10.6f}'
        print(string)

    def integrate_volume(
        self,
        X_in = None,
        θ_in = None,
    ):

        cstr = CSTReactor(
            microkin = self.microkin,
            reactor_volume = self.cstr_volume,
            vol_flow_rate = self.vol_flow_rate,
            t_bound = np.inf,
            method = self.method_cstr,
            rtol = self.rtol_cstr,
            atol = self.atol_cstr,
            store_output = False,
            dense_output = False,
        )

        X_in = np.array(self.microkin.X_list) if X_in is None else np.array(X_in)
        θ_in = np.array(self.microkin.θ_list) if θ_in is None else np.array(θ_in)
        X_zero = X_in.copy()
        θ_zero = θ_in.copy()
        
        if self.store_output is True:
            self.time_vect = [0.]
            self.X_vect = [X_in]
            self.θ_vect = [θ_in]
        
        if self.print_output is True:
            self.print_X_and_θ(
                z_reactor = 0.,
                X_array = X_in,
                θ_array = θ_in,
                print_names = True,
            )
        
        for ii in range(self.n_cstr):
            
            z_reactor = (ii+1)*self.cstr_length
            
            X_out, θ_out = cstr.integrate_volume(
                X_in = X_in,
                θ_in = θ_in,
                X_zero = X_zero,
                θ_zero = θ_zero,
            )
            
            # As first guess of the concentrations we assume a similar gas
            # phase conversion as the previous step.
            X_zero = X_out.copy()+(X_out-X_in)
            θ_zero = θ_out.copy()
            
            # The gas inlet of the next CSTR control volume is equal to the 
            # outlet of the current CSTR control volume.
            X_in = X_out.copy()
            
            if self.store_output is True:
                self.time_vect.append((ii+1)*self.cstr_volume/self.vol_flow_rate)
                self.X_vect.append(X_out)
                self.θ_vect.append(θ_out)
            
            # Print to screen the gas composition.
            if self.print_output is True:
                print_int = int(self.n_cstr/self.n_print) if self.n_cstr > 1 else 1
                if print_int > 0 and (ii+1) % print_int == 0:
                    self.print_X_and_θ(
                        z_reactor = z_reactor,
                        X_array = X_out,
                        θ_array = θ_out,
                        print_names = False,
                    )

        return X_out, θ_out

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
