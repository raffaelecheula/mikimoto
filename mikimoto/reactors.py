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
        t_bound = 1e+9,
        maxerr_dconc_dt = 1e-6,
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
        self.maxerr_dconc_dt = maxerr_dconc_dt
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
        
        if self.store_output is True:
            self.conc_vect = [y0]
            self.time_vect = [t0]
            self.temperature_vect = [self.microkin.temperature]
            self.pressure_vect = [self.microkin.pressure]
            self.X_vect = [self.microkin.X_list.copy()]
            self.θ_vect = [self.microkin.θ_list.copy()]
        if self.dense_output is True:
            interp_vect = []
        solver = self.initialize_solver(
            fun = fun,
            t0 = t0,
            y0 = y0,
        )
        n_steps = 0
        n_fails = 0
        self.max_dconc_dt = np.inf
        while (
            solver.status != 'finished' 
            and self.max_dconc_dt > self.maxerr_dconc_dt
            and n_steps < self.n_steps_max
        ):
            try: solver.step()
            except: pass
            if solver.status == 'failed':
                n_fails += 1
                y0 = solver.y + np.random.random(len(solver.y)) * 1e-6
                solver = self.initialize_solver(
                    fun = fun,
                    t0 = solver.t,
                    y0 = y0,
                )
            else:
                n_steps += 1
                if self.store_output is True:
                    self.conc_vect.append(solver.y)
                    self.time_vect.append(solver.t)
                    self.temperature_vect.append(self.microkin.temperature)
                    self.pressure_vect.append(self.microkin.pressure)
                    self.X_vect.append(self.microkin.X_list.copy())
                    self.θ_vect.append(self.microkin.θ_list.copy())
                if self.dense_output is True:
                    interp_vect.append(solver.dense_output())
            if n_fails > self.n_fails_max:
                raise RuntimeError("Calculation Failed!")
        if self.dense_output is True:
            self.conc_out_fun = OdeSolution(self.time_vect, interp_vect)
    
        return solver.y

    def integrate_volume(self):
        pass


class CSTReactor(IdealReactor):
    
    def __init__(
        self,
        microkin,
        reactor_volume = None,
        vol_flow_rate = None,
        contact_time = None,
        update_kinetics = False,
        method = 'LSODA',
        rtol = 1e-10,
        atol = 1e-12,
        t_bound = np.inf,
        n_steps_max = np.inf,
        n_fails_max = 1e3,
        maxerr_dconc_dt = 1e-6,
        store_output = True,
        dense_output = False,
    ):
        self.microkin = microkin
        self.update_kinetics = update_kinetics
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.t_bound = t_bound # [s]
        self.maxerr_dconc_dt = maxerr_dconc_dt
        self.n_steps_max = n_steps_max
        self.n_fails_max = n_fails_max
        self.store_output = store_output
        self.dense_output = dense_output
        # We use contact_freq instead of contact_time to avoid divisions by zero
        # for cases with no flow rates (e.g., batch reactors).
        if contact_time is not None:
            self.contact_freq = 1/contact_time # [1/s]
        elif reactor_volume is not None and vol_flow_rate is not None:
            self.contact_freq = vol_flow_rate/reactor_volume # [1/s]
        else:
            raise RuntimeError(
                "contact_time or reactor_volume and vol_flow_rate must be specified."
            )
    
    def integrate_volume(
        self,
        X_in = None, # [-]
        θ_in = None, # [-]
        X_zero = None, # [-]
        θ_zero = None, # [-]
    ):
        
        # dni_dt = ni_dot_in - ni_dot_out + sum(Rij)*volume
        # where ni are the number of moles, ni_dot_in and ni_dot_out
        # are the fluxes of moles entering and exiting the control volume
        # and are equal to the molar concentration entering and exiting
        # multiplied by the volumetric flow rate. Dividing by volume we get:
        # dCi_dt = (Ci_in - Ci_out)*vol_flow_rate/volume + sum(Rij)
        # (Ci = Pi/Rgas/T, Ci_in = conc_zero, Ci_out = conc).
        
        X_in = np.array(self.microkin.X_list) if X_in is None else np.array(X_in)
        θ_in = np.array(self.microkin.θ_list) if θ_in is None else np.array(θ_in)
        X_and_θ_in = np.concatenate([X_in, θ_in])
        conc_in = np.multiply(X_and_θ_in, self.microkin.conc_tot_species)
        
        X_zero = X_in.copy() if X_zero is None else np.array(X_zero)
        θ_zero = θ_in.copy() if θ_zero is None else np.array(θ_zero)
        X_and_θ_zero = np.concatenate([X_zero, θ_zero])
        conc_zero = np.multiply(X_and_θ_zero, self.microkin.conc_tot_species)
        
        def fun_dconc_dt(time, conc):
            if self.update_kinetics is True:
                self.microkin.get_kinetic_constants()
            self.microkin.get_reaction_rates(conc)
            dconc_dt = np.dot(
                self.microkin.stoich_coeffs.T, self.microkin.rates_net
            )
            dconc_dt += self.contact_freq*(conc_in-conc)*self.microkin.is_gas_spec
            self.max_dconc_dt = np.max(dconc_dt)
            return dconc_dt
        
        # Solve the system of differential equations.
        self.microkin.get_kinetic_constants()
        conc_out = self.integrate_ode(
            fun = fun_dconc_dt,
            t0 = 0.,
            y0 = conc_zero.copy(),
        )
        
        X_and_θ_out = np.divide(conc_out, self.microkin.conc_tot_species)
        X_out, θ_out = np.split(X_and_θ_out, [self.microkin.n_gas_species])
        
        self.microkin.X_list = list(X_out)
        self.microkin.θ_list = list(θ_out)
        
        return X_out, θ_out


class BatchReactor(CSTReactor):

    def __init__(
        self,
        microkin,
        reactor_volume,
        update_kinetics = False,
        method = 'LSODA',
        rtol = 1e-6,
        atol = 1e-8,
        t_bound = np.inf,
        n_steps_max = np.inf,
        n_fails_max = 1e3,
        maxerr_dconc_dt = 1e-6,
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
            maxerr_dconc_dt = maxerr_dconc_dt,
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
        maxerr_dconc_dt = 0.,
        n_steps_max = np.inf,
        n_fails_max = 1e3,
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
        self.maxerr_dconc_dt = maxerr_dconc_dt
        self.n_steps_max = n_steps_max
        self.n_fails_max = n_fails_max
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
        store_output = True,
        print_output = True,
        n_print = 10,
    ):
        self.microkin = microkin
        self.n_cstr = n_cstr
        self.reactor_volume = reactor_volume # [m^3]
        self.reactor_length = reactor_length # [m]
        self.vol_flow_rate = vol_flow_rate # [m^3/s]
        self.cstr_length = reactor_length/n_cstr # [m]
        self.cstr_volume = reactor_volume/n_cstr # [m^3]
        self.store_output = store_output
        self.print_output = print_output
        self.n_print = n_print

    def print_X_and_θ(
        self,
        z_reactor,
        X_array,
        θ_array,
        print_names = False,
        print_θ = False,
    ):
        if print_names is True:
            string = 'distance[m]'.rjust(14)
            for spec in self.microkin.gas_species:
                string += ('X_'+spec.name+'[-]').rjust(12)
            if print_θ is True:
                for spec in self.microkin.surf_species:
                    string += ('θ_'+spec.name+'[-]').rjust(12)
            print(string)
        string = f'  {z_reactor:12f}'
        for ii, spec in enumerate(self.microkin.gas_species):
            string += f'  {X_array[ii]:10.6f}'
        if print_θ is True:
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
