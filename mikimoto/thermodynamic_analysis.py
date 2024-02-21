# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
from scipy.optimize import root
from mikimoto import units
from mikimoto.utilities import NameAnalyzer

# -----------------------------------------------------------------------------
# GET NONEQUILIBRIUM RATIO
# -----------------------------------------------------------------------------

def get_nonequilirium_ratio(
    molfracs_dict,
    molfracs_eq_dict,
    reactants,
    products,
):
    
    eta_react = 1.
    for spec in products:
        eta_react *= molfracs_dict[spec]/molfracs_eq_dict[spec]
    for spec in reactants:
        eta_react *= molfracs_eq_dict[spec]/molfracs_dict[spec]

    return eta_react

# -----------------------------------------------------------------------------
# GET MOLFRACS FROM LAMBDA
# -----------------------------------------------------------------------------

def get_molfracs_from_lambda(
    molfracs_zero_dict,
    lambda_list,
    reactants_dict,
    products_dict,
):
    
    molfracs_dict = molfracs_zero_dict.copy()
    for spec in molfracs_dict:
        for ii, reaction in enumerate(reactants_dict):
            for spec_react in [
                spec_react for spec_react in reactants_dict[reaction]
                if spec_react == spec
            ]:
                molfracs_dict[spec] -= lambda_list[ii]
            for spec_prod in [
                spec_prod for spec_prod in products_dict[reaction]
                if spec_prod == spec
            ]:
                molfracs_dict[spec] += lambda_list[ii]

    molfracs_tot = sum([molfracs_dict[spec] for spec in molfracs_dict])
    for spec in molfracs_dict:
        molfracs_dict[spec] /= molfracs_tot

    return molfracs_dict

# -----------------------------------------------------------------------------
# ERROR ETA DICT
# -----------------------------------------------------------------------------

def error_eta_dict(
    x,
    eta_dict,
    molfracs_zero_dict,
    molfracs_eq_dict,
    reactants_dict,
    products_dict,
):
    
    molfracs_dict = get_molfracs_from_lambda(
        molfracs_zero_dict = molfracs_zero_dict,
        lambda_list = x,
        reactants_dict = reactants_dict,
        products_dict = products_dict,
    )

    error = []
    for react_name in eta_dict:
        reactants = reactants_dict[react_name]
        products = products_dict[react_name]
        eta_react = get_nonequilirium_ratio(
            molfracs_dict = molfracs_dict,
            molfracs_eq_dict = molfracs_eq_dict,
            reactants = reactants,
            products = products,
        )
        error.append(eta_react-eta_dict[react_name])

    for spec in molfracs_dict:
        if molfracs_dict[spec] < 0.:
            error += (0.-molfracs_dict[spec])*1e-3
        if molfracs_dict[spec] > 1.:
            error += (molfracs_dict[spec]-1.)*1e-3

    return error

# -----------------------------------------------------------------------------
# GET MOLFRACS FROM ETA DICT
# -----------------------------------------------------------------------------

def get_molfracs_from_eta_dict(
    gas,
    eta_dict,
    molfracs_zero_dict,
    x0 = None,
    method = 'hybr',
    options = None,
    name_analyzer = NameAnalyzer(),
):

    # TODO: change this
    gas.TPX = gas.T, gas.P, molfracs_zero_dict
    gas.equilibrate('TP')
    
    molfracs_eq_dict = {}
    for spec in gas.species_names:
        molfracs_eq_dict[spec] = gas[spec].X[0]
    
    reactants_dict = {}
    products_dict = {}
    for react_name in eta_dict:
        reactants_dict[react_name] = name_analyzer.get_reactants(react_name)
        products_dict[react_name] = name_analyzer.get_products(react_name)

    if x0 is None:
        x0 = [0.01]*len(eta_dict)
    
    results = root(
        fun = error_eta_dict,
        x0 = x0,
        method = method,
        options = options,
        args = (
            eta_dict,
            molfracs_zero_dict,
            molfracs_eq_dict,
            reactants_dict,
            products_dict,
        ),
    )
    if results.success is False:
        raise RuntimeError("root calculation not converged!")
    
    lambda_list = results.x
    
    molfracs_dict = get_molfracs_from_lambda(
        molfracs_zero_dict = molfracs_zero_dict,
        lambda_list = lambda_list,
        reactants_dict = reactants_dict,
        products_dict = products_dict,
    )

    gas.TPX = gas.T, gas.P, molfracs_dict

    return molfracs_dict

# -----------------------------------------------------------------------------
# GET DELTA G REACTION
# -----------------------------------------------------------------------------

def get_deltaG_reaction(
    reaction_name,
    gibbs_dict,
    name_analyzer = NameAnalyzer(),
):
    
    reactants = name_analyzer.get_reactants(reaction_name)
    products = name_analyzer.get_products(reaction_name)
    deltaG = 0.
    for spec in reactants:
        deltaG -= gibbs_dict[spec]
    for spec in products:
        deltaG += gibbs_dict[spec]

    return deltaG

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
