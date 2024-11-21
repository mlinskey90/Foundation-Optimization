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
    Calculate H'.
    Formula: H' = (2 * (M_z_ULS * γ_f) / Leff) + sqrt((F_Res_ULS * γ_f)^2 + (2 * (M_z_ULS * γ_f) / Leff)^2)
    """
    try:
        term1 = (2 * (M_z_ULS * load_factor_gamma_f)) / Leff
        term2 = np.sqrt((F_Res_ULS * load_factor_gamma_f) ** 2 + term1 ** 2)
        H_prime = term1 + term2
        print(f"Step 8: H' = {H_prime:.4f} kN")
        return H_prime
    except ZeroDivisionError:
        print("Step 8: Error - Leff is zero, cannot calculate H'.")
        return None
    except Exception as ex:
        print(f"Step 8: Error calculating H': {ex}")
        return None


def calculate_Madd(H_prime: float, h_water: float) -> Optional[float]:
    """
    Calculate Madd.
    Formula: Madd = H' * h_water
    """
    try:
        Madd = H_prime * h_water
        print(f"Step 9: Madd = {Madd:.4f} kNm")
        return Madd
    except Exception as ex:
        print(f"Step 9: Error calculating Madd: {ex}")
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
    print(f"Step 10: Mres = {Mres:.4f} kNm")
    return Mres


def calculate_sigma_net(sigma_xxf: Optional[float], e: float, d1: float, adjacent_stress: Optional[float]) -> Optional[float]:
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


def calculate_mc(sigma_net: Optional[float], Dx: float) -> Optional[float]:
    """
    Calculate Mc.
    Formula: Mc = (sigma_net * Dx^2) / 2 if σ_net > 0.
    """
    if sigma_net is None or sigma_net <= 0:
        print("Step 21: Mc = N/A (σ_net <= 0)")
        return None
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

    # Calculate bending moment
    Vd = calculate_Vd(total_weight, B_wet, sample_params.Fz_ULS)
    e = calculate_eccentricity(sample_params.MRes_without_Vd, Vd=Vd)
    A_eff = calculate_A_eff(sample_params.d1, e)
    B_e = calculate_B_e(sample_params.d1, e)
    L_e = calculate_L_e(sample_params.d1, B_e)
    Leff = calculate_Leff(A_eff, L_e, B_e)
    Beff = calculate_Beff(Leff, B_e)
    H_prime = calculate_H_prime(sample_params.M_z_ULS, sample_params.F_Res_ULS, Leff)
    Madd = calculate_Madd(H_prime, h_water=5.0)  # Assuming h_water = 5.0 m
    Mres = calculate_Mres(Madd, sample_params.MRes_without_Vd)
    Dx = (sample_params.d1 - sample_params.d2) / 2
    sigma_net = calculate_sigma_net(H_prime, e, sample_params.d1, adjacent_stress=None)  # Adjacent stress as None for now
    Mc = calculate_mc(sigma_net, Dx)
    wb = calculate_wb(sample_params.d1, sample_params.d2)
    Mt = calculate_mt(Mc, wb)

    # Output the final result
    if Mt is not None:
        print(f"\nFinal Output: Calculated Bending Moment (Mt) = {Mt:.2f} kNm")
    else:
        print("\nFinal Output: Bending Moment (Mt) could not be calculated due to insufficient data or 'N/A' conditions.")
