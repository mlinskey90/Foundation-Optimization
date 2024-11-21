import numpy as np

def calculate_volumes(params):
    """Calculate volumes of different parts of the foundation."""
    d1, d2, h1, h2, h3, h4, h5, b1, b2 = params
    C1 = (np.pi * d1**2 / 4) * h1
    C2 = (1 / 3) * np.pi * ((d1**2) + (d1 * d2) + (d2**2)) * h2
    C3 = (np.pi * d2**2 / 4) * h3
    C4 = (1 / 3) * np.pi * ((b1**2) + (b1 * b2) + (b2**2)) * h5
    return C1, C2, C3, C4

def calculate_weights(params, rho_conc):
    """Calculate total weight of the foundation."""
    C1, C2, C3, C4 = calculate_volumes(params)
    total_weight = (C1 + C2 + C3 + C4) * rho_conc
    return total_weight, C1, C2, C3, C4

def calculate_ballast_and_buoyancy(params, C2, C4, rho_ballast_wet, rho_water, rho_ballast_dry):
    """Calculate wet ballast, dry ballast, and buoyancy forces."""
    d1, d2, h1, h2, h3, h4 = params[:6]
    h_water = h1 + h2 + h3 - h4
    B_wet = ((np.pi * d1**2 / 4) * (h2 + h3 - h4) - C2 - (np.pi * d2**2 / 4) * (h3 - h4)) * rho_ballast_wet
    B_dry = ((np.pi * d1**2 / 4) * (h2 + h3 - h4) - C2 - (np.pi * d2**2 / 4) * (h3 - h4)) * rho_ballast_dry
    W = (((np.pi * d1**2 / 4) * h_water) + C4) * rho_water
    return B_wet, B_dry, W

def net_vertical_load(params, F_z, rho_conc, rho_ballast_wet, rho_water, rho_ballast_dry):
    """Calculate the net vertical load on the foundation."""
    total_weight, C1, C2, C3, C4 = calculate_weights(params, rho_conc)
    B_wet, B_dry, W = calculate_ballast_and_buoyancy(params, C2, C4, rho_ballast_wet, rho_water, rho_ballast_dry)
    net_load = W + B_wet + total_weight + F_z
    return net_load, total_weight, B_wet, B_dry, W

def calculate_pressures(params, F_z, F_RES, M_RES, rho_conc, rho_ballast_wet, rho_water, rho_ballast_dry):
    """Calculate minimum and maximum pressures at the base."""
    d1 = params[0]
    vertical_load, total_weight, B_wet, B_dry, W = net_vertical_load(params, F_z, rho_conc, rho_ballast_wet, rho_water, rho_ballast_dry)
    resultant_moment = M_RES + F_RES * sum(params[2:5])
    p_min = (vertical_load / (np.pi * d1**2 / 4)) - (resultant_moment / (np.pi * d1**3 / 32))
    p_max = (vertical_load / (np.pi * d1**2 / 4)) + (resultant_moment / (np.pi * d1**3 / 32))
    return p_min, p_max, B_wet, B_dry, W, vertical_load, total_weight
