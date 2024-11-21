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

def calculate_Beff(Leff: float, B_e: float, L_e: float) -> Optional[float]:
    """
    Calculate Beff.
    Formula: Beff = Leff * B_e / L_e
    """
    if L_e == 0:
        print("Step 7: L_e is zero, cannot calculate Beff.")
        return None  # Avoid division by zero
    try:
        Beff = (Leff * B_e) / L_e
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

def calculate_Mc(sigma_net: float, Dx: float, e: float, d1: float) -> Optional[float]:
    """
    Calculate Mc using the formula from the Excel sheet:
    IF(H146<=0,"N/A",IF(ISNUMBER(H144),IF(H121>=G33/2,"N/A",(H146*H143^2)/2),"N/A"))
    """
    try:
        if sigma_net <= 0:
            print("Step 9: sigma_net is less than or equal to 0, cannot calculate Mc.")
            return None
        if e >= d1 / 2:
            print("Step 9: Eccentricity (e) is greater than or equal to half of d1, cannot calculate Mc.")
            return None
        Mc = (sigma_net * Dx ** 2) / 2
        print(f"Step 9: Mc = {Mc:.4f} kNm")
        return Mc
    except Exception as ex:
        print(f"Step 9: Error calculating Mc: {ex}")
        return None

def calculate_wb(d1: float, d2: float) -> Optional[float]:
    """
    Calculate wb using the formula from the Excel sheet:
    wb = 2 * SQRT((d1/2)^2 - (d2/2)^2)
    """
    try:
        if d1 <= d2:
            print("Step 10: d1 must be greater than d2 to calculate wb.")
            return None
        wb = 2 * np.sqrt((d1 / 2) ** 2 - (d2 / 2) ** 2)
        print(f"Step 10: wb = {wb:.4f} m")
        return wb
    except Exception as ex:
        print(f"Step 10: Error calculating wb: {ex}")
        return None

def calculate_Mt(params: BendingMomentParams, total_weight: float, B_wet: float, sigma_net: float, Dx: float) -> Optional[float]:
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
    
    # Step 9: Calculate Mc
    Mc = calculate_Mc(sigma_net=sigma_net, Dx=Dx, e=e, d1=params.d1)
    if Mc is None:
        print("Calculation halted: Mc is None.")
        return None  # Cannot proceed without Mc
    
    # Step 10: Calculate wb
    wb = calculate_wb(d1=params.d1, d2=params.d2)
    if wb is None:
        print("Calculation halted: wb is None.")
        return None  # Cannot proceed without wb
    
    # Step 23: Calculate Mt
    try:
        Mt = Mc * wb
        print(f"Step 23: Mt = {Mt:.4f} kNm")
        return Mt
    except Exception as ex:
        print(f"Step 23: Error calculating Mt: {ex}")
        return None

# Example usage
if __name__ == "__main__":
    # Sample input values (replace these with actual data from your Streamlit app)
    sample_params = BendingMomentParams(
        Fz_ULS=6708.01,                 # Fz_ULS (kN)
        MRes_without_Vd=151200.0,        # MRes without Vd (kNm)
        d1=28
