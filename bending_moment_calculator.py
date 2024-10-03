# bending_moment_calculator.py

import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class BendingMomentParams:
    """
    Data class to hold all necessary input parameters for bending moment calculation.
    """
    Fz_ULS: float                     # Fz value for the Extreme Load Case (kN)
    load_factor_gamma_f: float        # Load factor γf
    MRes_without_Vd: float            # MRes without Vd (kNm)
    safety_factor_favorable: float    # Factor of safety for favorable conditions (e.g., 0.9)
    d1: float                         # Outer diameter (m)
    d2: float                         # Plinth diameter (m)
    h1: float                         # Specific height parameter h1 (m)
    h2: float                         # Specific height parameter h2 (m)
    h3: float                         # Specific height parameter h3 (m)
    h4: float                         # Specific height parameter h4 (m)
    h5: float                         # Specific height parameter h5 (m)
    b1: float                         # Specific breadth parameter b1 (m)
    b2: float                         # Specific breadth parameter b2 (m)
    rho_conc: float                   # Concrete density (kN/m³)
    rho_ballast_wet: float            # Ballast density (wet) (kN/m³)

def calculate_Vd(Fz_ULS: float,
               load_factor_gamma_f: float,
               MRes_without_Vd: float,
               safety_factor_favorable: float,
               total_weight: float,
               B_wet: float) -> float:
    """
    Calculate Vd.
    Formula: Vd = (total_weight + B_wet + Fz_ULS) * safety_factor_favorable
    """
    return (total_weight + B_wet + Fz_ULS) * safety_factor_favorable

def calculate_eccentricity(MRes_without_Vd: float,
                          load_factor_gamma_f: float,
                          Vd: float) -> Optional[float]:
    """
    Calculate eccentricity (e).
    Formula: e = (MRes_without_Vd * load_factor_gamma_f) / Vd
    """
    if Vd == 0:
        return None
    return (MRes_without_Vd * load_factor_gamma_f) / Vd

def calculate_A_eff(d1: float, e: float) -> Optional[float]:
    """
    Calculate A_eff (Effective foundation area).
    Formula: A_eff = 2 * [(0.25 * d1^2) * acos(e / (0.5 * d1)) - e * sqrt((0.25 * d1^2) - e^2)]
    """
    try:
        ratio = e / (0.5 * d1)
        if not -1 <= ratio <= 1:
            return None  # Invalid input for arccos
        term1 = 0.25 * d1**2 * np.arccos(ratio)
        term2 = e * np.sqrt((0.25 * d1**2) - e**2)
        return 2 * (term1 - term2)
    except:
        return None

def calculate_B_e(d1: float, e: float) -> Optional[float]:
    """
    Calculate B_e (Ellipse minor axis).
    Formula: B_e = 2 * (0.5 * d1 - e)
    """
    return 2 * (0.5 * d1 - e)

def calculate_L_e(d1: float, B_e: float) -> Optional[float]:
    """
    Calculate L_e (Ellipse major axis).
    Formula: L_e = 2 * (0.5 * d1) * sqrt(1 - (1 - (B_e / d1))^2)
    """
    try:
        ratio = 1 - (B_e / d1)
        inside_sqrt = 1 - ratio**2
        if inside_sqrt < 0:
            return None  # Invalid input for sqrt
        return 2 * (0.5 * d1) * np.sqrt(inside_sqrt)
    except:
        return None

def calculate_Leff(A_eff: float, L_e: float, B_e: float) -> Optional[float]:
    """
    Calculate Leff.
    Formula: Leff = sqrt(A_eff * L_e / B_e)
    """
    if B_e == 0:
        return None  # Avoid division by zero
    try:
        return np.sqrt((A_eff * L_e) / B_e)
    except:
        return None

def calculate_Beff(Leff: float, Be: float, Le: float) -> Optional[float]:
    """
    Calculate Beff.
    Formula: Beff = Leff * Be / Le
    """
    if Le == 0:
        return None  # Avoid division by zero
    try:
        return (Leff * Be) / Le
    except:
        return None

# Placeholder for H'
def calculate_H_prime(...) -> Optional[float]:
    """
    Calculate H'.
    Formula: [Provide the formula here]
    """
    # TODO: Implement the formula for H'
    return None

# Placeholder for Madd
def calculate_Madd(...) -> Optional[float]:
    """
    Calculate Madd.
    Formula: [Provide the formula here]
    """
    # TODO: Implement the formula for Madd
    return None

# Placeholder for Fxy
def calculate_Fxy(...) -> Optional[float]:
    """
    Calculate Fxy.
    Formula: [Provide the formula here]
    """
    # TODO: Implement the formula for Fxy
    return None

# Placeholder for Fz
def calculate_Fz(...) -> Optional[float]:
    """
    Calculate Fz.
    Formula: [Provide the formula here]
    """
    # TODO: Implement the formula for Fz
    return None

def calculate_L_over_6(Leff: float, Beff: float) -> float:
    """
    Calculate L_over_6.
    Formula: L_over_6 = MAX(Leff, Beff) / 6
    """
    return max(Leff, Beff) / 6

def calculate_sigma_max(e: float, L_over_6: float, Vd: float,
                       Leff: float, Beff: float) -> Optional[float]:
    """
    Calculate sigma_max.
    Formula:
    IF(e > L_over_6,
        (2 * Vd) / ((3 * MIN(Leff, Beff)) * (0.5 * MAX(Leff, Beff) - e)),
        (Vd / (Leff * Beff)) + (6 * Vd) / (MIN(Leff, Beff) * MAX(Leff, Beff)^2))
    """
    min_Leff_Beff = min(Leff, Beff)
    max_Leff_Beff = max(Leff, Beff)
    if e > L_over_6:
        denominator = (3 * min_Leff_Beff) * (0.5 * max_Leff_Beff - e)
        if denominator == 0:
            return None  # Avoid division by zero
        try:
            return (2 * Vd) / denominator
        except:
            return None
    else:
        denominator1 = Leff * Beff
        denominator2 = min_Leff_Beff * (max_Leff_Beff ** 2)
        if denominator1 == 0 or denominator2 == 0:
            return None  # Avoid division by zero
        try:
            return (Vd / denominator1) + ((6 * Vd) / denominator2)
        except:
            return None

def calculate_sigma_min(e: float, L_over_6: float, Vd: float,
                       Leff: float, Beff: float, Le: float,
                       Mres: float) -> Optional[float]:
    """
    Calculate sigma_min.
    Formula:
    IF(e > L_over_6, 0, (Vd / (Leff * Beff)) - (6 * Mres) / (MIN(Leff, Beff) * (MAX(Leff, Beff)^2)))
    """
    if e > L_over_6:
        return 0
    else:
        denominator1 = Leff * Beff
        min_Leff_Beff = min(Leff, Beff)
        max_Leff_Beff = max(Leff, Beff)
        if denominator1 == 0 or (min_Leff_Beff * (max_Leff_Beff ** 2)) == 0:
            return None  # Avoid division by zero
        try:
            return (Vd / denominator1) - ((6 * Mres) / (min_Leff_Beff * (max_Leff_Beff ** 2)))
        except:
            return None

def calculate_Lp(e: float, L_over_6: float, Lp_formula_params) -> Optional[float]:
    """
    Calculate Lp.
    Formula: IF(e > L_over_6, 3 * ((MAX(d1, d2) / 2) - e), "N/A")
    """
    if e > L_over_6:
        max_diameter = max(Lp_formula_params['d1'], Lp_formula_params['d2'])
        return 3 * ((max_diameter / 2) - e)
    else:
        return None  # "N/A"

def calculate_Dx(d1: float, d2: float) -> float:
    """
    Calculate Dx.
    Formula: Dx = (d1 - d2) / 2
    """
    return (d1 - d2) / 2

def calculate_sigma_xxf(e: float, L_over_6: float, Lp: Optional[float],
                       Dx: float, sigma_max: Optional[float], sigma_min: Optional[float],
                       d1: float) -> Optional[float]:
    """
    Calculate sigma_xxf.
    Formula:
    IF(e >= d1/2, "N/A",
        IF(e > L_over_6,
            IF(Lp < Dx, "N/A-Lp<Dx", sigma_max - ((sigma_max - sigma_min) / Lp) * Dx),
            sigma_max - ((sigma_max - sigma_min) / d1) * Dx))
    """
    if e >= d1 / 2:
        return None  # "N/A"
    if e > L_over_6:
        if Lp is None or Dx == 0:
            return None  # "N/A-Lp<Dx"
        if Lp < Dx:
            return None  # "N/A-Lp<Dx"
        if Lp == 0:
            return None  # Avoid division by zero
        if sigma_max is None or sigma_min is None:
            return None
        try:
            return sigma_max - ((sigma_max - sigma_min) / Lp) * Dx
        except:
            return None
    else:
        if d1 == 0:
            return None  # Avoid division by zero
        if sigma_max is None or sigma_min is None:
            return None
        try:
            return sigma_max - ((sigma_max - sigma_min) / d1) * Dx
        except:
            return None

def calculate_sigma_adjacent(e: float, d1: float, d2: float,
                            h1: float, h2: float, h3: float,
                            safety_factor_favorable: float,
                            rho_conc: float, rho_ballast_wet: float) -> Optional[float]:
    """
    Calculate Adjacent Stress (H145).
    Formula:
    IF(e >= d1/2, "N/A",
        IF((h2 + h3) < h2,
            (h1 + h2) * rho_conc * safety_factor_favorable,
            (h1 + h2) * rho_conc * safety_factor_favorable + (h1 + h2 - 0.5 * h2) * rho_ballast_wet * safety_factor_favorable))
    """
    if e >= d1 / 2:
        return None  # "N/A"
    if (h2 + h3) < h2:
        return (h1 + h2) * rho_conc * safety_factor_favorable
    else:
        return ((h1 + h2) * rho_conc * safety_factor_favorable) + ((h1 + h2) - 0.5 * h2) * rho_ballast_wet * safety_factor_favorable

def calculate_sigma_net(sigma_xxf: Optional[float], e: float,
                       d1: float, adjacent_stress: Optional[float]) -> Optional[float]:
    """
    Calculate sigma_net.
    Formula: IF(ISNUMBER(sigma_xxf), IF(e >= d1/2, "N/A", sigma_xxf - adjacent_stress), "N/A")
    """
    if isinstance(sigma_xxf, (int, float)):
        if e >= d1 / 2:
            return None  # "N/A"
        else:
            if adjacent_stress is None:
                return sigma_xxf  # If adjacent_stress is "N/A", assume it's zero or handle accordingly
            return sigma_xxf - adjacent_stress
    else:
        return None  # "N/A"

def calculate_mc(sigma_net: Optional[float], Dx: float) -> Optional[float]:
    """
    Calculate Mc.
    Formula: Mc = (sigma_net * Dx^2) / 2
    """
    if sigma_net is None:
        return None  # Cannot calculate Mc if sigma_net is "N/A"
    try:
        return (sigma_net * (Dx ** 2)) / 2
    except:
        return None

def calculate_wb(d1: float, d2: float) -> Optional[float]:
    """
    Calculate wb.
    Formula: wb = 2 * sqrt((d1/2)^2 - (d2/2)^2)
    """
    term = (d1 / 2) ** 2 - (d2 / 2) ** 2
    if term < 0:
        return None  # To avoid sqrt of negative number
    try:
        return 2 * np.sqrt(term)
    except:
        return None

def calculate_mt(mc: Optional[float], wb: Optional[float]) -> Optional[float]:
    """
    Calculate Mt.
    Formula: Mt = Mc * wb
    """
    if mc is None or wb is None:
        return None  # Cannot calculate Mt
    try:
        return mc * wb
    except:
        return None

def calculate_Fxy(...) -> Optional[float]:
    """
    Calculate Fxy.
    Formula: [Provide the formula here]
    """
    # TODO: Implement the formula for Fxy
    return None

def calculate_Fz(...) -> Optional[float]:
    """
    Calculate Fz.
    Formula: [Provide the formula here]
    """
    # TODO: Implement the formula for Fz
    return None

# Placeholder for sigma_x-xsc * FOSfavorable
def calculate_sigma_x_xsc_FOSfavorable(...) -> Optional[float]:
    """
    Calculate sigma_x-xsc * FOSfavorable.
    Formula: [Provide the formula here]
    """
    # TODO: Implement the formula for sigma_x-xsc * FOSfavorable
    return None

def calculate_bending_moment(params: BendingMomentParams, total_weight: float, B_wet: float) -> Optional[float]:
    """
    Main function to calculate the bending moment Mt = Mc * wb.
    It sequentially calculates all intermediate values based on the provided parameters.
    """
    # Step 1: Calculate Vd
    Vd = calculate_Vd(
        Fz_ULS=params.Fz_ULS,
        load_factor_gamma_f=params.load_factor_gamma_f,
        MRes_without_Vd=params.MRes_without_Vd,
        safety_factor_favorable=params.safety_factor_favorable,
        total_weight=total_weight,
        B_wet=B_wet
    )
    if Vd is None:
        return None  # Cannot proceed without Vd
    
    # Step 2: Calculate e (eccentricity)
    e = calculate_eccentricity(
        MRes_without_Vd=params.MRes_without_Vd,
        load_factor_gamma_f=params.load_factor_gamma_f,
        Vd=Vd
    )
    if e is None:
        return None  # Cannot proceed without e
    
    # Step 3: Calculate A_eff
    A_eff = calculate_A_eff(d1=params.d1, e=e)
    if A_eff is None:
        return None  # Cannot proceed without A_eff
    
    # Step 4: Calculate B_e
    B_e = calculate_B_e(d1=params.d1, e=e)
    if B_e is None:
        return None  # Cannot proceed without B_e
    
    # Step 5: Calculate L_e
    L_e = calculate_L_e(d1=params.d1, B_e=B_e)
    if L_e is None:
        return None  # Cannot proceed without L_e
    
    # Step 6: Calculate Leff
    Leff = calculate_Leff(A_eff=A_eff, L_e=L_e, B_e=B_e)
    if Leff is None:
        return None  # Cannot proceed without Leff
    
    # Step 7: Calculate Beff
    # Le is assumed to be 0.6 meters (H125) as it's not part of BendingMomentParams
    Le = 0.6
    Beff = calculate_Beff(Leff=Leff, Be=B_e, Le=Le)
    if Beff is None:
        return None  # Cannot proceed without Beff
    
    # Step 8: Calculate H' (Placeholder)
    H_prime = calculate_H_prime(...)  # Replace ... with actual parameters when available
    # TODO: Implement H_prime calculation
    # For now, we'll skip using H_prime until the formula is provided
    
    # Step 9: Calculate Madd (Placeholder)
    Madd = calculate_Madd(...)  # Replace ... with actual parameters when available
    # TODO: Implement Madd calculation
    
    # Step 10: Calculate Mres (Already provided as MRes_without_Vd)
    Mres = params.MRes_without_Vd
    
    # Step 11: Calculate Fxy (Placeholder)
    Fxy = calculate_Fxy(...)  # Replace ... with actual parameters when available
    # TODO: Implement Fxy calculation
    
    # Step 12: Calculate Fz (Placeholder)
    Fz = calculate_Fz(...)  # Replace ... with actual parameters when available
    # TODO: Implement Fz calculation
    
    # Step 13: Calculate L/6
    L_over_6 = calculate_L_over_6(Leff=Leff, Beff=Beff)
    
    # Step 14: Calculate sigma_max
    sigma_max = calculate_sigma_max(
        e=e,
        L_over_6=L_over_6,
        Vd=Vd,
        Leff=Leff,
        Beff=Beff
    )
    if sigma_max is None:
        return None  # Cannot proceed without sigma_max
    
    # Step 15: Calculate sigma_min
    sigma_min = calculate_sigma_min(
        e=e,
        L_over_6=L_over_6,
        Vd=Vd,
        Leff=Leff,
        Beff=Beff,
        Le=Le,
        Mres=Mres
    )
    if sigma_min is None and e <= L_over_6:
        return None  # Cannot proceed without sigma_min
    
    # Step 16: Calculate Lp
    Lp = calculate_Lp(
        e=e,
        L_over_6=L_over_6,
        Lp_formula_params={'d1': params.d1, 'd2': params.d2}
    )
    # Lp is None if "N/A"
    
    # Step 17: Calculate Dx
    Dx = calculate_Dx(d1=params.d1, d2=params.d2)
    
    # Step 18: Calculate sigma_x-xf
    sigma_xxf = calculate_sigma_xxf(
        e=e,
        L_over_6=L_over_6,
        Lp=Lp,
        Dx=Dx,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        d1=params.d1
    )
    if sigma_xxf is None:
        return None  # Cannot proceed without sigma_x-xf
    
    # Step 19: Calculate sigma_x-xsc * FOSfavorable (Placeholder)
    sigma_x_xsc_FOSfavorable = calculate_sigma_x_xsc_FOSfavorable(...)  # Replace ... with actual parameters
    # TODO: Implement sigma_x-xsc * FOSfavorable calculation
    
    # Step 20: Calculate sigma_net
    sigma_net = calculate_sigma_net(
        sigma_xxf=sigma_xxf,
        e=e,
        d1=params.d1,
        adjacent_stress=None  # Will be calculated next
    )
    # Note: Adjust 'adjacent_stress' as per your requirements
    
    if sigma_net is None:
        return None  # Cannot proceed without sigma_net
    
    # Step 21: Calculate Mc
    Mc = calculate_mc(
        sigma_net=sigma_net,
        Dx=Dx
    )
    if Mc is None:
        return None  # Cannot proceed without Mc
    
    # Step 22: Calculate wb
    wb = calculate_wb(
        d1=params.d1,
        d2=params.d2
    )
    if wb is None:
        return None  # Cannot proceed without wb
    
    # Step 23: Calculate Mt
    Mt = calculate_mt(
        mc=Mc,
        wb=wb
    )
    if Mt is None:
        return None  # Cannot proceed without Mt
    
    return Mt

# Example Usage
if __name__ == "__main__":
    # Sample input values (replace these with actual data from your Streamlit app)
    sample_params = BendingMomentParams(
        Fz_ULS=3300.0,                 # Fz_ULS (kN)
        load_factor_gamma_f=1.2,        # Load factor γf
        MRes_without_Vd=1000.0,         # MRes without Vd (kNm)
        safety_factor_favorable=0.9,    # Safety factor (e.g., 0.9)
        d1=10.0,                        # Outer diameter (m)
        d2=6.0,                         # Plinth diameter (m)
        h1=1.0,                         # Specific height parameter h1 (m)
        h2=2.0,                         # Specific height parameter h2 (m)
        h3=1.5,                         # Specific height parameter h3 (m)
        h4=0.1,                         # Specific height parameter h4 (m)
        h5=0.25,                        # Specific height parameter h5 (m)
        b1=6.0,                         # Specific breadth parameter b1 (m)
        b2=5.5,                         # Specific breadth parameter b2 (m)
        rho_conc=24.5,                   # Concrete density (kN/m³)
        rho_ballast_wet=20.0             # Ballast density (wet) (kN/m³)
    )
    
    # Assuming total_weight and B_wet are calculated elsewhere (e.g., via Streamlit functions)
    total_weight = 1000.0  # Example value (kN)
    B_wet = 2000.0          # Example value (kN)
    
    # Calculate bending moment
    Mt = calculate_bending_moment(sample_params, total_weight, B_wet)
    
    if Mt is not None:
        print(f"Calculated Bending Moment (Mt): {Mt:.2f} kNm")
    else:
        print("Bending Moment (Mt) could not be calculated due to insufficient data or 'N/A' conditions.")
