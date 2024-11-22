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


def calculate_Beff(Leff: float, Be: float, Le: float) -> Optional[float]:
    """
    Calculate Beff.
    Formula: Beff = Leff * Be / Le
    """
    # Debug input values
    print(f"Inputs to calculate_Beff: Leff = {Leff:.4f}, B_e = {Be:.4f}, Le = {Le:.4f}")
    
    # Validate Le to avoid division by zero or invalid values
    if Le <= 0:
        print(f"Step 7: Invalid Le value ({Le:.4f}). Le must be greater than zero.")
        return None
    
    try:
        # Calculate Beff
        Beff = (Leff * Be) / Le
        print(f"Step 7: Beff = {Beff:.4f} m")
        return Beff
    except Exception as ex:
        print(f"Step 7: Error calculating Beff: {ex}")
        return None

def calculate_Dx(d1: float, d2: float) -> float:
    """
    Calculate Dx.
    Formula: Dx = (d1 - d2) / 2
    """
    Dx = (d1 - d2) / 2
    print(f"Step 8: Dx = {Dx:.4f} m")
    return Dx

def calculate_L_sixth(Leff: float, B_e: float) -> float:
    """
    Calculate L/6.
    Formula: L_sixth = max(Leff, B_e) / 6
    """
    L_sixth = max(Leff, B_e) / 6
    print(f"Step 9: L_sixth = {L_sixth:.4f} m")
    return L_sixth



def calculate_H_prime(M_z_ULS: float, F_Res_ULS: float, Leff: float, load_factor_gamma_f: float = 1.0) -> Optional[float]:
    """
    Calculate H'.
    Formula: H' = (2 * (M_z_ULS * γ_f) / Leff) + sqrt((F_Res_ULS * γ_f)^2 + (2 * (M_z_ULS * γ_f) / Leff)^2)
    """
    try:
        term1 = (2 * (M_z_ULS * load_factor_gamma_f)) / Leff
        term2 = np.sqrt((F_Res_ULS * load_factor_gamma_f) ** 2 + term1 ** 2)
        H_prime = term1 + term2
        print(f"Step 10: H' = {H_prime:.4f} kN")
        return H_prime
    except ZeroDivisionError:
        print("Step 10: Error - Leff is zero, cannot calculate H'.")
        return None
    except Exception as ex:
        print(f"Step 10: Error calculating H': {ex}")
        return None


def calculate_Madd(H_prime: float, h_water: float) -> Optional[float]:
    """
    Calculate Madd.
    Formula: Madd = H' * h_water
    """
    try:
        Madd = H_prime * h_water
        print(f"Step 11: Madd = {Madd:.4f} kNm")
        return Madd
    except Exception as ex:
        print(f"Step 11: Error calculating Madd: {ex}")
        return None


def calculate_Fxy(H_prime: float) -> Optional[float]:
    """
    Calculate Fxy.
    Formula: Fxy = H'
    """
    return H_prime


def calculate_Fz(Fz_ULS: float, load_factor_gamma_f: float = 1.0) -> Optional[float]:
    """
    Calculate Fz.
    Formula: Fz = Fz_ULS * γ_f
    """
    Fz = Fz_ULS * load_factor_gamma_f
    print(f"Step 12: Fz = {Fz:.4f} kN")
    return Fz


def calculate_Mres(Madd: float, MRes_ULS: float, load_factor_gamma_f: float = 1.0) -> Optional[float]:
    """
    Calculate Mres.
    Formula: Mres = Madd + (MRes_ULS * γ_f)
    """
    Mres = Madd + (MRes_ULS * load_factor_gamma_f)
    print(f"Step 12: Mres = {Mres:.4f} kNm")
    return Mres

def calculate_sigma_max(Vd: float, Leff: float, Beff: float, eccentricity: float, L_sixth: float) -> float:
    """
    Calculate σmax.
    Formula: 
    - If eccentricity > L/6:
        σmax = (2 * Vd) / ((3 * min(Leff, Beff)) * (0.5 * max(Leff, Beff) - eccentricity))
    - Otherwise:
        σmax = (Vd / (Leff * Beff)) + ((6 * Vd) / (min(Leff, Beff) * (max(Leff, Beff)^2)))
    """
    min_dim = min(Leff, Beff)
    max_dim = max(Leff, Beff)

    print(f"Intermediate: min_dim = {min_dim:.4f}, max_dim = {max_dim:.4f}, L_sixth = {L_sixth:.4f}")
    print(f"Debug: Checking condition: eccentricity = {eccentricity:.4f}, L_sixth = {L_sixth:.4f}")

    if eccentricity > L_sixth:
        # Debugging intermediate calculations for Case 1
        print("Debug: Branch 1 (eccentricity > L_sixth)")
        denominator = (3 * min_dim) * (0.5 * max_dim - eccentricity)
        if denominator <= 0:
            raise ValueError(f"Invalid denominator in σmax calculation: {denominator}")
        sigma_max = (2 * Vd) / denominator
        print(f"Case 1: σmax = (2 * {Vd:.4f}) / ({denominator:.4f}) = {sigma_max:.4f}")
    else:
        # Debugging intermediate calculations for Case 2
        print("Debug: Branch 2 (eccentricity <= L_sixth)")
        term1 = Vd / (Leff * Beff)
        term2_denominator = min_dim * max_dim**2
        if term2_denominator <= 0:
            raise ValueError(f"Invalid denominator in σmax Case 2 term 2: {term2_denominator}")
        term2 = (6 * Vd) / term2_denominator
        sigma_max = term1 + term2
        print(f"Case 2: σmax = ({term1:.4f}) + ({term2:.4f}) = {sigma_max:.4f}")

    print(f"Step 13: σmax = {sigma_max:.4f} kN/m²")
    return sigma_max

def calculate_sigma_min(
    Vd: float,
    Leff: float,
    Beff: float,
    Mres: float,
    L_sixth: float,
    eccentricity: float
) -> float:
    """
    Calculate σmin.
    Formula:
    - If eccentricity > L_sixth, σmin = 0.
    - Otherwise:
        σmin = (Vd / (Leff * Beff)) - ((6 * Mres) / (min(Leff, Beff) * (max(Leff, Beff)^2))).
    """
    if eccentricity > L_sixth:
        print("Step 13: σmin = 0 (eccentricity > L_sixth)")
        return 0

    min_dim = min(Leff, Beff)
    max_dim = max(Leff, Beff)
    sigma_min = (Vd / (Leff * Beff)) - ((6 * Mres) / (min_dim * max_dim**2))
    print(f"Step 13: σmin = {sigma_min:.4f} kN/m²")
    return sigma_min


def calculate_Lp(
    Leff: float,
    Beff: float,
    eccentricity: float,
    L_sixth: float
) -> Optional[float]:
    """
    Calculate Lp.
    Formula:
    - If eccentricity > L_sixth:
        Lp = 3 * ((max(Leff, Beff) / 2) - eccentricity).
    - Otherwise, return None ("N/A").
    """
    if eccentricity > L_sixth:
        max_dim = max(Leff, Beff)
        Lp = 3 * ((max_dim / 2) - eccentricity)
        print(f"Step 14: Lp = {Lp:.4f} m")
        return Lp
    else:
        print("Step 14: Lp = N/A (eccentricity <= L_sixth)")
        return None


def calculate_sigma_xxf(
    sigma_max: float,
    sigma_min: float,
    eccentricity: float,
    d1: float,
    L_sixth: float,
    Lp: float,
    Dx: float
) -> Optional[float]:
    """
    Calculate σx-xf (stress due to worst forces).
    Formula:
    - If eccentricity >= d1 / 2, return None ("N/A").
    - If eccentricity > L_sixth:
        - If Lp < Dx, return None ("N/A-Lp<Dx").
        - Otherwise, σx-xf = σmax - ((σmax - σmin) / Lp) * Dx.
    - Otherwise:
        σx-xf = σmax - ((σmax - σmin) / d1) * Dx.
    """
    if eccentricity >= d1 / 2:
        print("Step 14: σx-xf = N/A (eccentricity >= d1 / 2)")
        return None

    if eccentricity > L_sixth:
        if Lp < Dx:
            print("Step 14: σx-xf = N/A-Lp<Dx (Lp < Dx)")
            return None
        sigma_xxf = sigma_max - ((sigma_max - sigma_min) / Lp) * Dx
    else:
        sigma_xxf = sigma_max - ((sigma_max - sigma_min) / d1) * Dx

    print(f"Step 14: σx-xf = {sigma_xxf:.4f} kN/m²")
    return sigma_xxf

def calculate_sigma_xsc_FOS_favourable(h1: float, h2: float, h3: float, rho_conc: float, rho_ballast_wet: float, FOS_favourable: float) -> float:
    """
    Calculate σx-xsc * FOS_favourable.
    Formula: 
    - If (h2 + h3) < h2:
        σx-xsc = ((h1 + h2) * rho_conc * FOS_favourable)
    - Otherwise:
        σx-xsc = ((h1 + h2) * rho_conc * FOS_favourable) + 
                 (((h1 + h2) - 0.5 * h2) * rho_ballast_wet * FOS_favourable)
    """
    if (h2 + h3) < h2:
        sigma_xsc = (h1 + h2) * rho_conc * FOS_favourable
    else:
        sigma_xsc = ((h1 + h2) * rho_conc * FOS_favourable) + (((h1 + h2) - 0.5 * h2) * rho_ballast_wet * FOS_favourable)
    print(f"Step 14: σx-xsc * FOS_favourable = {sigma_xsc:.4f} kN/m²")
    return sigma_xsc

  
def calculate_sigma_net(
    sigma_xxf: Optional[float],
    sigma_xsc_favourable: float
) -> Optional[float]:
    """
    Calculate σ_net.
    Formula: σ_net = σx-xf - σx-xsc * FOS_favourable.
    """
    if sigma_xxf is None:
        print("Step 15: σ_net = N/A (σx-xf is None)")
        return None

    sigma_net = sigma_xxf - sigma_xsc_favourable
    print(f"Step 15: σ_net = {sigma_net:.4f} kN/m²")
    return sigma_net


def calculate_mc(sigma_net: Optional[float], Dx: float) -> Optional[float]:
    """
    Calculate Mc.
    Formula: Mc = (sigma_net * Dx^2) / 2 if σ_net > 0.
    """
    if sigma_net is None or sigma_net <= 0:
        print("Step 16: Mc = N/A (σ_net <= 0)")
        return None
    try:
        Mc = (sigma_net * (Dx ** 2)) / 2
        print(f"Step 16: Mc = {Mc:.4f} kNm")
        return Mc
    except Exception as ex:
        print(f"Step 16: Error calculating Mc: {ex}")
        return None


def calculate_wb(d1: float, d2: float) -> Optional[float]:
    """
    Calculate wb.
    Formula: wb = 2 * sqrt((d1/2)^2 - (d2/2)^2)
    """
    term = (d1 / 2) ** 2 - (d2 / 2) ** 2
    if term < 0:
        print("Step 17: Invalid term for sqrt in wb calculation. wb = N/A")
        return None  # To avoid sqrt of negative number
    try:
        wb = 2 * np.sqrt(term)
        print(f"Step 17: wb = {wb:.4f} m")
        return wb
    except Exception as ex:
        print(f"Step 17: Error calculating wb: {ex}")
        return None


def calculate_mt(mc: Optional[float], wb: Optional[float]) -> Optional[float]:
    """
    Calculate Mt.
    Formula: Mt = Mc * wb
    """
    if mc is None or wb is None:
        print("Step 18: Mt = N/A (Mc or wb is None)")
        return None  # Cannot calculate Mt
    try:
        Mt = mc * wb
        print(f"Step 18: Mt = {Mt:.4f} kNm")
        return Mt
    except Exception as ex:
        print(f"Step 18: Error calculating Mt: {ex}")
        return None


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
        M_z_ULS=18613.0,                 # Moment in z-direction at ULS (kNm)
        F_Res_ULS=1271.0                 # Resisting force at ULS (kN)
    )

    # Assuming total_weight and B_wet are calculated elsewhere
    total_weight = 21434.94  # Total weight (kN)
    B_wet = 26753.17          # Wet ballast force (kN)
    FOS_favourable = 0.9     # Favourable Factor of Safety

    # Calculate bending moment
    Vd = calculate_Vd(total_weight, B_wet, sample_params.Fz_ULS)
    e = calculate_eccentricity(sample_params.MRes_without_Vd, Vd=Vd)
    A_eff = calculate_A_eff(sample_params.d1, e)
    B_e = calculate_B_e(sample_params.d1, e)
    L_e = calculate_L_e(sample_params.d1, B_e)
    Leff = calculate_Leff(A_eff, L_e, B_e)
    Beff = calculate_Beff(Leff, B_e, L_e)
    Dx = calculate_Dx(sample_params.d1, sample_params.d2)
    L_sixth = calculate_L_sixth(Leff, B_e)
    H_prime = calculate_H_prime(sample_params.M_z_ULS, sample_params.F_Res_ULS, Leff)
    Madd = calculate_Madd(H_prime, h_water=3.55)  # Assuming h_water = 3.55 m
    Mres = calculate_Mres(Madd, sample_params.MRes_without_Vd)
    sigma_max = calculate_sigma_max(Vd, Leff, Beff, e, L_sixth)
    Lp = calculate_Lp(Leff, Beff, e, L_sixth)
    sigma_min = calculate_sigma_min(Vd, Leff, Beff, Mres, L_sixth, e)
    sigma_xxf = calculate_sigma_xxf(sigma_max, sigma_min, e, sample_params.d1, L_sixth, Lp if Lp is not None else sample_params.d1, Dx)
    sigma_xsc_favourable = calculate_sigma_xsc_FOS_favourable(
        sample_params.h1, sample_params.h2, sample_params.h3, sample_params.rho_conc, sample_params.rho_ballast_wet, FOS_favourable
    )
    sigma_net = calculate_sigma_net(sigma_xxf, sigma_xsc_favourable)
    Mc = calculate_mc(sigma_net, Dx)
    wb = calculate_wb(sample_params.d1, sample_params.d2)
    Mt = calculate_mt(Mc, wb)

    # Output the final result
    if Mt is not None:
        print(f"\nFinal Output: Calculated Bending Moment (Mt) = {Mt:.2f} kNm")
    else:
        print("\nFinal Output: Bending Moment (Mt) could not be calculated due to insufficient data or 'N/A' conditions.")
