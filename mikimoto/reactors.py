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
        rtol = 1e-8,
        atol = 1e-10,
        t_bound = 1e+9,
        norm_dconc_thr = 1e-9,
        n_steps_max = np.inf,
        n_fails_max = 1e3,
        dense_output = False,
    ):
        self.microkin = microkin
        self.update_kinetics = update_kinetics
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.t_bound = t_bound
        self.norm_dconc_thr = norm_dconc_thr
        self.n_steps_max = n_steps_max
        self.n_fails_max = n_fails_max
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
        
        conc_vect = [y0]
        time_vect = [t0]
        interp_vect = []
        solver = self.initialize_solver(
            fun = fun,
            t0 = t0,
            y0 = y0,
        )
        n_steps = 0
        n_fails = 0
        self.norm_dconc = np.inf
        while (
            solver.status != 'finished' 
            and self.norm_dconc > self.norm_dconc_thr
            and n_fails < self.n_fails_max
            and n_steps < self.n_steps_max
        ):
            solver.step()
            if solver.status == 'failed':
                n_fails += 1
                solver = self.initialize_solver(
                    fun = fun,
                    t0 = solver.t,
                    y0 = solver.y,
                )
                continue
            n_steps += 1
            conc_vect.append(solver.y)
            time_vect.append(solver.t)
            interp_vect.append(solver.dense_output())
        self.conc_vect = conc_vect
        self.time_vect = time_vect
        if self.dense_output is True:
            self.conc_out_fun = OdeSolution(time_vect, interp_vect)
    
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
        rtol = 1e-6,
        atol = 1e-8,
        t_bound = np.inf,
        n_steps_max = np.inf,
        n_fails_max = 1e3,
        norm_dconc_thr = 1e-9,
        dense_output = False,
    ):
        self.microkin = microkin
        self.update_kinetics = update_kinetics
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.t_bound = t_bound # [s]
        self.norm_dconc_thr = norm_dconc_thr
        self.n_steps_max = n_steps_max
        self.n_fails_max = n_fails_max
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
        x_in, # [-]
        θ_in, # [-]
        x_zero = None, # [-]
        θ_zero = None, # [-]
    ):
        
        # dni_dt = ni_dot_in - ni_dot_out + sum(Rij)*volume
        # where ni are the number of moles, ni_dot_in and ni_dot_out
        # are the fluxes of moles entering and exiting the control volume
        # and are equal to the molar concentration entering and exiting
        # multiplied by the volumetric flow rate. Dividing by volume we get:
        # dCi_dt = (Ci_in - Ci_out)*vol_flow_rate/volume + sum(Rij)
        # (Ci = Pi/Rgas/T, Ci_in = conc_zero, Ci_out = conc).
        
        x_in = np.array(x_in)
        θ_in = np.array(θ_in)
        x_and_θ_in = np.concatenate([x_in, θ_in])
        conc_in = np.multiply(x_and_θ_in, self.microkin.conc_tot_species)
        
        x_zero = x_in.copy() if x_zero is None else np.array(x_zero)
        θ_zero = θ_in.copy() if θ_zero is None else np.array(θ_zero)
        x_and_θ_zero = np.concatenate([x_zero, θ_zero])
        conc_zero = np.multiply(x_and_θ_zero, self.microkin.conc_tot_species)
        
        def fun_dconc_dt(time, conc):
            if self.update_kinetics is True:
                self.microkin.get_kinetic_constants()
            self.microkin.get_reaction_rates(conc)
            dconc_dt = np.dot(
                self.microkin.stoich_coeffs.T, self.microkin.rates_net
            )
            dconc_dt += self.contact_freq*(conc_in-conc)*self.microkin.is_gas_spec
            self.norm_dconc = np.linalg.norm(dconc_dt)/len(conc)
            return dconc_dt
        
        # Solve the system of differential equations.
        self.microkin.get_kinetic_constants()
        conc_out = self.integrate_ode(
            fun = fun_dconc_dt,
            t0 = 0.,
            y0 = conc_zero.copy(),
        )
        
        x_and_θ_out = np.divide(conc_out, self.microkin.conc_tot_species)
        x_out, θ_out = np.split(x_and_θ_out, [self.microkin.n_gas_species])
        
        return x_out, θ_out


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
        norm_dconc_thr = 1e-9,
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
            norm_dconc_thr = norm_dconc_thr,
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
        rtol = 1e-4,
        atol = 1e-6,
        norm_dconc_thr = 0.,
        n_steps_max = np.inf,
        n_fails_max = 1e3,
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
        self.norm_dconc_thr = norm_dconc_thr
        self.n_steps_max = n_steps_max
        self.n_fails_max = n_fails_max
        self.dense_output = dense_output
        self.cstr_volume = vol_flow_rate*delta_time # [m^3]
        self.t_bound = reactor_volume/vol_flow_rate # [s]

    def integrate_volume(
        self,
        x_in, # [-]
        θ_in, # [-]
    ):

        x_in = np.array(x_in)
        θ_in = np.array(θ_in)
        
        cstr = CSTReactor(
            microkin = self.microkin,
            reactor_volume = self.cstr_volume,
            vol_flow_rate = self.vol_flow_rate,
            t_bound = 1e+9,
            dense_output = False,
        )

        self.x_zero = x_in.copy()
        self.θ_zero = θ_in.copy()
        def fun_dconc_dt(time, conc_gas):
            x_in = conc_gas/self.microkin.conc_tot_gas
            x_out, θ_out = cstr.integrate_volume(
                x_in = x_in,
                θ_in = θ_in,
                x_zero = self.x_zero,
                θ_zero = self.θ_zero,
            )
            self.x_zero = x_out.copy()
            self.θ_zero = θ_out.copy()
            dconc_dt = (x_out-x_in)*self.microkin.conc_tot_gas/self.delta_time
            return dconc_dt
        
        # Solve the system of differential equations.
        self.microkin.get_kinetic_constants()
        conc_out = self.integrate_ode(
            fun = fun_dconc_dt,
            t0 = 0.,
            y0 = x_in*self.microkin.conc_tot_gas,
        )
        
        x_out = conc_out/self.microkin.conc_tot_gas
        
        return x_out


class PFReactorSeriesCSTR:
    def __init__(
        self,
        microkin,
        n_cstr,
        reactor_volume,
        reactor_length,
        vol_flow_rate,
        n_print = 10,
    ):
        self.microkin = microkin
        self.n_cstr = n_cstr
        self.reactor_volume = reactor_volume # [m^3]
        self.reactor_length = reactor_length # [m]
        self.vol_flow_rate = vol_flow_rate # [m^3/s]
        self.cstr_length = reactor_length/n_cstr # [m]
        self.cstr_volume = reactor_volume/n_cstr # [m^3]
        self.n_print = n_print

    def print_x_and_θ(
        self,
        z_reactor,
        x_array,
        θ_array,
        print_names = False,
        print_θ = True,
    ):
        if print_names is True:
            string = 'distance[m]'.rjust(14)
            for spec in self.microkin.gas_species:
                string += ('x_'+spec.name+'[-]').rjust(12)
            if print_θ is True:
                for spec in self.microkin.surf_species:
                    string += ('θ_'+spec.name+'[-]').rjust(12)
            print(string)
        string = f'  {z_reactor:12f}'
        for ii, spec in enumerate(self.microkin.gas_species):
            string += f'  {x_array[ii]:10.6f}'
        if print_θ is True:
            for ii, spec in enumerate(self.microkin.surf_species):
                string += f'  {θ_array[ii]:10.6f}'
        print(string)

    def integrate_volume(
        self,
        x_in,
        θ_in,
    ):

        cstr = CSTReactor(
            microkin = self.microkin,
            reactor_volume = self.cstr_volume,
            vol_flow_rate = self.vol_flow_rate,
            t_bound = np.inf,
            dense_output = False,
        )

        x_in = np.array(x_in)
        θ_in = np.array(θ_in)
        x_zero = x_in.copy()
        θ_zero = θ_in.copy()
        
        self.print_x_and_θ(
            z_reactor = 0.,
            x_array = x_in,
            θ_array = θ_in,
            print_names = True,
        )
        
        for ii in range(self.n_cstr):
            
            z_reactor = (ii+1)*self.cstr_length
            
            x_out, θ_out = cstr.integrate_volume(
                x_in = x_in,
                θ_in = θ_in,
                x_zero = x_zero,
                θ_zero = θ_zero,
            )
            
            # As first guess of the concentrations we assume a similar gas
            # phase conversion as the previous step.
            x_zero = x_out.copy()+(x_out-x_in)
            θ_zero = θ_out.copy()
            
            # The gas inlet of the next CSTR control volume is equal to the 
            # outlet of the current CSTR control volume.
            x_in = x_out.copy()
            
            # Print to screen the gas composition.
            print_int = int(self.n_cstr/self.n_print) if self.n_cstr > 1 else 1
            if print_int > 0 and (ii+1) % print_int == 0:
                self.print_x_and_θ(
                    z_reactor = z_reactor,
                    x_array = x_out,
                    θ_array = θ_out,
                    print_names = False,
                )

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
