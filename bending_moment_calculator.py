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
    MRes_without_Vd: float            # MRes without Vd (kNm)
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
    M_z_ULS: float                    # Moment in z-direction at ULS (kNm)
    F_Res_ULS: float                  # Resisting force at ULS (kN)

def calculate_Vd(total_weight: float, B_wet: float, Fz_ULS: float, safety_factor_favorable: float = 0.9) -> float:
    """
    Calculate Vd.
    Formula: Vd = (total_weight + B_wet + Fz_ULS) * safety_factor_favorable
    """
    Vd = (total_weight + B_wet + Fz_ULS) * safety_factor_favorable
    print(f"Step 1: Vd = {Vd:.2f} kN")
    return Vd

def calculate_eccentricity(MRes_without_Vd: float, load_factor_gamma_f: float = 1.0, Vd: float = 0.0) -> Optional[float]:
    """
    Calculate eccentricity (e).
    Formula: e = (MRes_without_Vd * load_factor_gamma_f) / Vd
    """
    if Vd == 0:
        print("Step 2: Vd is zero, cannot calculate eccentricity (e).")
        return None
    e = (MRes_without_Vd * load_factor_gamma_f) / Vd
    print(f"Step 2: Eccentricity (e) = {e:.4f} m")
    return e

def calculate_A_eff(d1: float, e: float) -> Optional[float]:
    """
    Calculate A_eff (Effective foundation area).
    Formula: A_eff = 2 * [(0.25 * d1^2) * acos(e / (0.5 * d1)) - e * sqrt((0.25 * d1^2) - e^2)]
    """
    try:
        ratio = e / (0.5 * d1)
        if not -1 <= ratio <= 1:
            print(f"Step 3: Invalid ratio for acos: {ratio:.4f}. Cannot calculate A_eff.")
            return None  # Invalid input for arccos
        term1 = 0.25 * d1**2 * np.arccos(ratio)
        term2 = e * np.sqrt((0.25 * d1**2) - e**2)
        A_eff = 2 * (term1 - term2)
        print(f"Step 3: A_eff = {A_eff:.4f} m²")
        return A_eff
    except Exception as ex:
        print(f"Step 3: Error calculating A_eff: {ex}")
        return None

def calculate_B_e(d1: float, e: float) -> Optional[float]:
    """
    Calculate B_e (Ellipse minor axis).
    Formula: B_e = 2 * (0.5 * d1 - e)
    """
    B_e = 2 * (0.5 * d1 - e)
    print(f"Step 4: B_e = {B_e:.4f} m")
    return B_e

def calculate_L_e(d1: float, B_e: float) -> Optional[float]:
    """
    Calculate L_e (Ellipse major axis).
    Formula: L_e = 2 * (0.5 * d1) * sqrt(1 - (1 - (B_e / d1))^2)
    """
    try:
        ratio = 1 - (B_e / d1)
        inside_sqrt = 1 - ratio**2
        if inside_sqrt < 0:
            print(f"Step 5: Invalid inside_sqrt value: {inside_sqrt:.4f}. Cannot calculate L_e.")
            return None  # Invalid input for sqrt
        L_e = 2 * (0.5 * d1) * np.sqrt(inside_sqrt)
        print(f"Step 5: L_e = {L_e:.4f} m")
        return L_e
    except Exception as ex:
        print(f"Step 5: Error calculating L_e: {ex}")
        return None

def calculate_Leff(A_eff: float, L_e: float, B_e: float) -> Optional[float]:
    """
    Calculate Leff.
    Formula: Leff = sqrt(A_eff * L_e / B_e)
    """
    if B_e == 0:
        print("Step 6: B_e is zero, cannot calculate Leff.")
        return None  # Avoid division by zero
    try:
        Leff = np.sqrt((A_eff * L_e) / B_e)
        print(f"Step 6: Leff = {Leff:.4f} m")
        return Leff
    except Exception as ex:
        print(f"Step 6: Error calculating Leff: {ex}")
        return None

def calculate_Beff(Leff: float, Be: float, Le: float = 0.6) -> Optional[float]:
    """
    Calculate Beff.
    Formula: Beff = Leff * Be / Le
    """
    if Le == 0:
        print("Step 7: Le is zero, cannot calculate Beff.")
        return None  # Avoid division by zero
    try:
        Beff = (Leff * Be) / Le
        print(f"Step 7: Beff = {Beff:.4f} m")
        return Beff
    except Exception as ex:
        print(f"Step 7: Error calculating Beff: {ex}")
        return None

def calculate_H_prime(M_z_ULS: float, F_Res_ULS: float, Leff: float, load_factor_gamma_f: float = 1.0) -> Optional[float]:
    """
    Calculate H' (Equivalent Horizontal Force).
    Formula: H_prime = (2 * (M_z_ULS * load_factor_gamma_f) / Leff) + 
                      sqrt((F_Res_ULS * load_factor_gamma_f)^2 + (2 * (M_z_ULS * load_factor_gamma_f) / Leff)^2)
    """
    try:
        term1 = (2 * (M_z_ULS * load_factor_gamma_f)) / Leff
        term2 = np.sqrt((F_Res_ULS * load_factor_gamma_f) ** 2 + (2 * (M_z_ULS * load_factor_gamma_f) / Leff) ** 2)
        H_prime = term1 + term2
        print(f"Step 8: H' = {H_prime:.4f} kN")
        return H_prime
    except ZeroDivisionError:
        print("Step 8: Error - Leff is zero, cannot calculate H'.")
        return None
    except Exception as ex:
        print(f"Step 8: Error calculating H': {ex}")
        return None

# Placeholder for Madd
def calculate_Madd(...) -> Optional[float]:
    """
    Calculate Madd.
    Formula: [Provide the formula here]
    """
    # TODO: Implement the formula for Madd
    print("Step 9: Madd calculation is not implemented yet.")
    return None

# Placeholder for Fxy
def calculate_Fxy(...) -> Optional[float]:
    """
    Calculate Fxy.
    Formula: [Provide the formula here]
    """
    # TODO: Implement the formula for Fxy
    print("Step 11: Fxy calculation is not implemented yet.")
    return None

# Placeholder for Fz
def calculate_Fz(...) -> Optional[float]:
    """
    Calculate Fz.
    Formula: [Provide the formula here]
    """
    # TODO: Implement the formula for Fz
    print("Step 12: Fz calculation is not implemented yet.")
    return None

def calculate_L_over_6(Leff: float, Beff: float) -> float:
    """
    Calculate L_over_6.
    Formula: L_over_6 = MAX(Leff, Beff) / 6
    """
    L_over_6 = max(Leff, Beff) / 6
    print(f"Step 13: L_over_6 = {L_over_6:.4f} m")
    return L_over_6

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
            print("Step 14: Denominator is zero in sigma_max calculation.")
            return None  # Avoid division by zero
        try:
            sigma_max = (2 * Vd) / denominator
            print(f"Step 14: sigma_max = {sigma_max:.4f} kN/m²")
            return sigma_max
        except Exception as ex:
            print(f"Step 14: Error calculating sigma_max: {ex}")
            return None
    else:
        denominator1 = Leff * Beff
        denominator2 = min_Leff_Beff * (max_Leff_Beff ** 2)
        if denominator1 == 0 or denominator2 == 0:
            print("Step 14: One of the denominators is zero in sigma_max calculation.")
            return None  # Avoid division by zero
        try:
            sigma_max = (Vd / denominator1) + ((6 * Vd) / denominator2)
            print(f"Step 14: sigma_max = {sigma_max:.4f} kN/m²")
            return sigma_max
        except Exception as ex:
            print(f"Step 14: Error calculating sigma_max: {ex}")
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
        sigma_min = 0
        print(f"Step 15: sigma_min = {sigma_min} kN/m²")
        return sigma_min
    else:
        denominator1 = Leff * Beff
        min_Leff_Beff = min(Leff, Beff)
        max_Leff_Beff = max(Leff, Beff)
        if denominator1 == 0 or (min_Leff_Beff * (max_Leff_Beff ** 2)) == 0:
            print("Step 15: Denominator is zero in sigma_min calculation.")
            return None  # Avoid division by zero
        try:
            sigma_min = (Vd / denominator1) - ((6 * Mres) / (min_Leff_Beff * (max_Leff_Beff ** 2)))
            print(f"Step 15: sigma_min = {sigma_min:.4f} kN/m²")
            return sigma_min
        except Exception as ex:
            print(f"Step 15: Error calculating sigma_min: {ex}")
            return None

def calculate_Lp(e: float, L_over_6: float, d1: float, d2: float) -> Optional[float]:
    """
    Calculate Lp.
    Formula: IF(e > L_over_6, 3 * ((MAX(d1, d2) / 2) - e), "N/A")
    """
    if e > L_over_6:
        max_diameter = max(d1, d2)
        Lp = 3 * ((max_diameter / 2) - e)
        print(f"Step 16: Lp = {Lp:.4f} m")
        return Lp
    else:
        print("Step 16: Lp = N/A")
        return None  # "N/A"

def calculate_Dx(d1: float, d2: float) -> float:
    """
    Calculate Dx.
    Formula: Dx = (d1 - d2) / 2
    """
    Dx = (d1 - d2) / 2
    print(f"Step 17: Dx = {Dx:.4f} m")
    return Dx

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
        print("Step 18: sigma_x-xf = N/A (e >= d1/2)")
        return None  # "N/A"
    if e > L_over_6:
        if Lp is None or Dx == 0:
            print("Step 18: sigma_x-xf = N/A-Lp<Dx")
            return None  # "N/A-Lp<Dx"
        if Lp < Dx:
            print("Step 18: sigma_x-xf = N/A-Lp<Dx")
            return None  # "N/A-Lp<Dx"
        if Lp == 0:
            print("Step 18: Lp is zero, cannot calculate sigma_x-xf.")
            return None  # Avoid division by zero
        if sigma_max is None or sigma_min is None:
            print("Step 18: sigma_max or sigma_min is None, cannot calculate sigma_x-xf.")
            return None
        try:
            sigma_xxf = sigma_max - ((sigma_max - sigma_min) / Lp) * Dx
            print(f"Step 18: sigma_x-xf = {sigma_xxf:.4f} kN/m²")
            return sigma_xxf
        except Exception as ex:
            print(f"Step 18: Error calculating sigma_x-xf: {ex}")
            return None
    else:
        if d1 == 0:
            print("Step 18: d1 is zero, cannot calculate sigma_x-xf.")
            return None  # Avoid division by zero
        if sigma_max is None or sigma_min is None:
            print("Step 18: sigma_max or sigma_min is None, cannot calculate sigma_x-xf.")
            return None
        try:
            sigma_xxf = sigma_max - ((sigma_max - sigma_min) / d1) * Dx
            print(f"Step 18: sigma_x-xf = {sigma_xxf:.4f} kN/m²")
            return sigma_xxf
        except Exception as ex:
            print(f"Step 18: Error calculating sigma_x-xf: {ex}")
            return None

def calculate_sigma_x_xsc_FOSfavorable(...) -> Optional[float]:
    """
    Calculate sigma_x-xsc * FOSfavorable.
    Formula: [Provide the formula here]
    """
    # TODO: Implement the formula for sigma_x-xsc * FOSfavorable
    print("Step 19: sigma_x-xsc * FOSfavorable calculation is not implemented yet.")
    return None

def calculate_sigma_net(sigma_xxf: Optional[float], e: float,
                       d1: float, adjacent_stress: Optional[float]) -> Optional[float]:
    """
    Calculate sigma_net.
    Formula: IF(ISNUMBER(sigma_xxf), IF(e >= d1/2, "N/A", sigma_xxf - adjacent_stress), "N/A")
    """
    if isinstance(sigma_xxf, (int, float)):
        if e >= d1 / 2:
            print("Step 20: sigma_net = N/A (e >= d1/2)")
            return None  # "N/A"
        else:
            if adjacent_stress is None:
                sigma_net = sigma_xxf  # If adjacent_stress is "N/A", assume it's zero or handle accordingly
                print(f"Step 20: sigma_net = {sigma_net:.4f} kN/m² (adjacent_stress = N/A)")
                return sigma_net
            sigma_net = sigma_xxf - adjacent_stress
            print(f"Step 20: sigma_net = {sigma_net:.4f} kN/m²")
            return sigma_net
    else:
        print("Step 20: sigma_xxf is not a number, sigma_net = N/A")
        return None  # "N/A"

def calculate_sigma_adjacent(e: float, d1: float, d2: float,
                            h1: float, h2: float, h3: float,
                            safety_factor_favorable: float,
                            rho_conc: float, rho_ballast_wet: float) -> Optional[float]:
    """
    Calculate Adjacent Stress (H').
    Formula:
    IF(e >= d1/2, "N/A",
        IF((h2 + h3) < h2,
            (h1 + h2) * rho_conc * safety_factor_favorable,
            (h1 + h2) * rho_conc * safety_factor_favorable + (h1 + h2 - 0.5 * h2) * rho_ballast_wet * safety_factor_favorable))
    """
    if e >= d1 / 2:
        print("Step 20: Adjacent Stress (H') = N/A (e >= d1/2)")
        return None  # "N/A"
    if (h2 + h3) < h2:
        adjacent_stress = (h1 + h2) * rho_conc * safety_factor_favorable
        print(f"Step 20: Adjacent Stress (H') = {adjacent_stress:.4f} kN/m²")
        return adjacent_stress
    else:
        adjacent_stress = ((h1 + h2) * rho_conc * safety_factor_favorable) + (((h1 + h2) - 0.5 * h2) * rho_ballast_wet * safety_factor_favorable)
        print(f"Step 20: Adjacent Stress (H') = {adjacent_stress:.4f} kN/m²")
        return adjacent_stress

def calculate_mc(sigma_net: Optional[float], Dx: float) -> Optional[float]:
    """
    Calculate Mc.
    Formula: Mc = (sigma_net * Dx^2) / 2
    """
    if sigma_net is None:
        print("Step 21: Mc = N/A (sigma_net is None)")
        return None  # Cannot calculate Mc if sigma_net is "N/A"
    try:
        Mc = (sigma_net * (Dx ** 2)) / 2
        print(f"Step 21: Mc = {Mc:.4f} kNm")
        return Mc
    except Exception as ex:
        print(f"Step 21: Error calculating Mc: {ex}")
        return None

def calculate_wb(d1: float, d2: float) -> Optional[float]:
    """
    Calculate wb.
    Formula: wb = 2 * sqrt((d1/2)^2 - (d2/2)^2)
    """
    term = (d1 / 2) ** 2 - (d2 / 2) ** 2
    if term < 0:
        print("Step 22: Invalid term for sqrt in wb calculation. wb = N/A")
        return None  # To avoid sqrt of negative number
    try:
        wb = 2 * np.sqrt(term)
        print(f"Step 22: wb = {wb:.4f} m")
        return wb
    except Exception as ex:
        print(f"Step 22: Error calculating wb: {ex}")
        return None

def calculate_mt(mc: Optional[float], wb: Optional[float]) -> Optional[float]:
    """
    Calculate Mt.
    Formula: Mt = Mc * wb
    """
    if mc is None or wb is None:
        print("Step 23: Mt = N/A (Mc or wb is None)")
        return None  # Cannot calculate Mt
    try:
        Mt = mc * wb
        print(f"Step 23: Mt = {Mt:.4f} kNm")
        return Mt
    except Exception as ex:
        print(f"Step 23: Error calculating Mt: {ex}")
        return None

def calculate_bending_moment(params: BendingMomentParams, total_weight: float, B_wet: float) -> Optional[float]:
    """
    Main function to calculate the bending moment Mt = Mc * wb.
    It sequentially calculates all intermediate values based on the provided parameters.
    """
    # Constants
    safety_factor_favorable = 0.9
    load_factor_gamma_f = 1.0
    
    # Step 1: Calculate Vd
    Vd = calculate_Vd(
        total_weight=total_weight,
        B_wet=B_wet,
        Fz_ULS=params.Fz_ULS,
        safety_factor_favorable=safety_factor_favorable
    )
    if Vd is None:
        print("Calculation halted: Vd is None.")
        return None  # Cannot proceed without Vd
    
    # Step 2: Calculate e (eccentricity)
    e = calculate_eccentricity(
        MRes_without_Vd=params.MRes_without_Vd,
        load_factor_gamma_f=load_factor_gamma_f,
        Vd=Vd
    )
    if e is None:
        print("Calculation halted: Eccentricity (e) is None.")
        return None  # Cannot proceed without e
    
    # Step 3: Calculate A_eff
    A_eff = calculate_A_eff(d1=params.d1, e=e)
    if A_eff is None:
        print("Calculation halted: A_eff is None.")
        return None  # Cannot proceed without A_eff
    
    # Step 4: Calculate B_e
    B_e = calculate_B_e(d1=params.d1, e=e)
    if B_e is None:
        print("Calculation halted: B_e is None.")
        return None  # Cannot proceed without B_e
    
    # Step 5: Calculate L_e
    L_e = calculate_L_e(d1=params.d1, B_e=B_e)
    if L_e is None:
        print("Calculation halted: L_e is None.")
        return None  # Cannot proceed without L_e
    
    # Step 6: Calculate Leff
    Leff = calculate_Leff(A_eff=A_eff, L_e=L_e, B_e=B_e)
    if Leff is None:
        print("Calculation halted: Leff is None.")
        return None  # Cannot proceed without Leff
    
    # Step 7: Calculate Beff
    # Le is assumed to be 0.6 meters (H125) as it's not part of BendingMomentParams
    Le = 0.6
    Beff = calculate_Beff(Leff=Leff, Be=B_e, Le=Le)
    if Beff is None:
        print("Calculation halted: Beff is None.")
        return None  # Cannot proceed without Beff
    
    # Step 8: Calculate H' (Equivalent Horizontal Force)
    H_prime = calculate_H_prime(
        M_z_ULS=params.M_z_ULS,
        F_Res_ULS=params.F_Res_ULS,
        Leff=Leff,
        load_factor_gamma_f=load_factor_gamma_f
    )
    if H_prime is None:
        print("Calculation halted: H' is None.")
        return None  # Cannot proceed without H'
    
    # Step 9: Calculate Madd (Placeholder)
    Madd = calculate_Madd(...)  # Replace ... with actual parameters when available
    # TODO: Implement Madd calculation
    
    # Step 10: Calculate Mres (Already provided as MRes_without_Vd)
    Mres = params.MRes_without_Vd
    print(f"Step 10: Mres = {Mres:.2f} kNm")
    
    # Step 11: Calculate Fxy (Placeholder)
    Fxy = calculate_Fxy(...)  # Replace ... with actual parameters when available
    
    # Step 12: Calculate Fz (Placeholder)
    Fz = calculate_Fz(...)  # Replace ... with actual parameters when available
    
    # Step 13: Calculate L_over_6
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
        print("Calculation halted: sigma_max is None.")
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
        print("Calculation halted: sigma_min is None.")
        return None  # Cannot proceed without sigma_min
    
    # Step 16: Calculate Lp
    Lp = calculate_Lp(
        e=e,
        L_over_6=L_over_6,
        d1=params.d1,
        d2=params.d2
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
        print("Calculation halted: sigma_x-xf is None.")
        return None  # Cannot proceed without sigma_x-xf
    
    # Step 19: Calculate sigma_x-xsc * FOSfavorable (Placeholder)
    sigma_x_xsc_FOSfavorable = calculate_sigma_x_xsc_FOSfavorable(...)  # Replace ... with actual parameters
    
    # Step 20: Calculate Adjacent Stress (H')
    adjacent_stress = calculate_sigma_adjacent(
        e=e,
        d1=params.d1,
        d2=params.d2,
        h1=params.h1,
        h2=params.h2,
        h3=params.h3,
        safety_factor_favorable=safety_factor_favorable,
        rho_conc=params.rho_conc,
        rho_ballast_wet=params.rho_ballast_wet
    )
    
    # Step 21: Calculate sigma_net
    sigma_net = calculate_sigma_net(
        sigma_xxf=sigma_xxf,
        e=e,
        d1=params.d1,
        adjacent_stress=adjacent_stress
    )
    if sigma_net is None:
        print("Calculation halted: sigma_net is None.")
        return None  # Cannot proceed without sigma_net
    
    # Step 22: Calculate Mc
    Mc = calculate_mc(
        sigma_net=sigma_net,
        Dx=Dx
    )
    if Mc is None:
        print("Calculation halted: Mc is None.")
        return None  # Cannot proceed without Mc
    
    # Step 23: Calculate wb
    wb = calculate_wb(
        d1=params.d1,
        d2=params.d2
    )
    if wb is None:
        print("Calculation halted: wb is None.")
        return None  # Cannot proceed without wb
    
    # Step 24: Calculate Mt
    Mt = calculate_mt(
        mc=Mc,
        wb=wb
    )
    if Mt is None:
        print("Calculation halted: Mt is None.")
        return None  # Cannot proceed without Mt
    
    return Mt

# Example Usage
if __name__ == "__main__":
    # Sample input values (replace these with actual data from your Streamlit app)
    sample_params = BendingMomentParams(
        Fz_ULS=6708.01,                 # Fz_ULS (kN)
        MRes_without_Vd=151200.0,        # MRes without Vd (kNm)
        d1=28.1,                         # Outer diameter (m)
        d2=6.5,                          # Plinth diameter (m)
        h1=0.4,                          # Specific height parameter h1 (m)
        h2=2.2,                          # Specific height parameter h2 (m)
        h3=1.05,                         # Specific height parameter h3 (m)
        h4=0.1,                          # Specific height parameter h4 (m)
        h5=0.25,                         # Specific height parameter h5 (m)
        b1=6.5,                          # Specific breadth parameter b1 (m)
        b2=6.0,                          # Specific breadth parameter b2 (m)
        rho_conc=24.5,                   # Concrete density (kN/m³)
        rho_ballast_wet=20.0,            # Ballast density (wet) (kN/m³)
        M_z_ULS=18613.0,                  # Moment in z-direction at ULS (kNm) - Default test value
        F_Res_ULS=1271.0                 # Resisting force at ULS (kN) - Default test value
    )
    
    # Assuming total_weight and B_wet are calculated elsewhere (e.g., via Streamlit functions)
    total_weight = 21434.94  # Total weight (kN)
    B_wet = 26753.17          # Wet ballast force (kN)
    
    # Calculate bending moment
    Mt = calculate_bending_moment(sample_params, total_weight, B_wet)
    
    # Output the final result
    if Mt is not None:
        print(f"\nFinal Output: Calculated Bending Moment (Mt) = {Mt:.2f} kNm")
    else:
        print("\nFinal Output: Bending Moment (Mt) could not be calculated due to insufficient data or 'N/A' conditions.")
