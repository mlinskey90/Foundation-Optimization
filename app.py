import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize

# Define the necessary functions
def calculate_foundation_weight(params, rho_conc):
    d1, d2, h1, h2, h3, h4, h5, b1, b2 = params
    C1 = (np.pi * d1**2 / 4) * h1 * rho_conc
    C2 = (1/3) * np.pi * ((d1/2)**2 + (d1/2 * d2/2) + (d2/2)**2) * h2 * rho_conc
    C3 = (np.pi * d2**2 / 4) * h3 * rho_conc
    C4 = (1/3) * np.pi * ((b1/2)**2 + (b1/2 * b2/2) + (b2/2)**2) * h5 * rho_conc
    total_weight = C1 + C2 + C3 + C4

    return total_weight, C1, C2, C3, C4

def calculate_ballast_and_buoyancy(params, C2, C4, rho_ballast_wet, rho_water, rho_conc):
    d1, d2, h1, h2, h3, h4, h5 = params[0], params[1], params[2], params[3], params[4], params[5], params[6]
    h_water = h1 + h2 + h3 - h4
    B_wet = ((np.pi * d1**2 / 4) * (h2 + h3 - h4) - (C2 / rho_conc) - (np.pi * d2**2 / 4) * (h3 - h4)) * rho_ballast_wet
    W = ((np.pi * d1**2 / 4) * h_water + (C4 / rho_conc)) * rho_water  # Corrected buoyancy force calculation
    return B_wet, W

def net_vertical_load(params, F_z, rho_conc, rho_ballast_wet, rho_water):
    total_weight, C1, C2, C3, C4 = calculate_foundation_weight(params, rho_conc)
    B_wet, W = calculate_ballast_and_buoyancy(params, C2, C4, rho_ballast_wet, rho_water, rho_conc)
    net_load = W + B_wet + C1 + C2 + C3 + C4 + F_z
    return net_load, total_weight, B_wet, W

def calculate_pressures(params, F_z, F_RES, M_RES, rho_conc, rho_ballast_wet, rho_water):
    d1 = params[0]
    vertical_load, total_weight, B_wet, W = net_vertical_load(params, F_z, rho_conc, rho_ballast_wet, rho_water)
    M_RES2 = M_RES + F_RES * (params[2] + params[3] + params[4])  # M_RES2 = M_RES + F_RES x (h1 + h2 + h3)
    resultant_moment = M_RES2

    p_min = (vertical_load / (np.pi * d1**2 / 4)) - (resultant_moment / (np.pi * d1**3 / 32))
    p_max = (vertical_load / (np.pi * d1**2 / 4)) + (resultant_moment / (np.pi * d1**3 / 32))

    return p_min, p_max, B_wet, W, vertical_load, total_weight

def plot_foundation_comparison(original_params, optimized_params):
    fig, ax = plt.subplots(figsize=(20, 15))

    def plot_foundation(params, edgecolor, fillcolor, label):
        d1, d2, h1, h2, h3, h4, h5, b1, b2 = params

        plinth_x = [-d2/2, d2/2, d2/2, -d2/2, -d2/2]
        plinth_y = [h1+h2+h3, h1+h2+h3, h1+h2, h1+h2, h1+h2+h3]

        haunch_x = [-d1/2, d1/2, d2/2, -d2/2, -d1/2]
        haunch_y = [h1, h1, h1+h2, h1+h2, h1]

        slab_x = [-d1/2, d1/2, d1/2, -d1/2, -d1/2]
        slab_y = [0, 0, h1, h1, 0]

        downstand_x = [-b1/2, -b2/2, b2/2, b1/2, -b1/2]
        downstand_y = [0, -h5, -h5, 0, 0]

        ax.plot(plinth_x, plinth_y, color=edgecolor)
        ax.plot(haunch_x, haunch_y, color=edgecolor)
        ax.plot(slab_x, slab_y, color=edgecolor)
        ax.plot(downstand_x, downstand_y, color=edgecolor)

        ax.fill(plinth_x, plinth_y, color=fillcolor, alpha=0.3, edgecolor=edgecolor, label=label)
        ax.fill(haunch_x, haunch_y, color=fillcolor, alpha=0.3, edgecolor=edgecolor)
        ax.fill(slab_x, slab_y, color=fillcolor, alpha=0.3, edgecolor=edgecolor)
        ax.fill(downstand_x, downstand_y, color=fillcolor, alpha=0.3, edgecolor=edgecolor)

    plot_foundation(original_params, 'black', 'grey', 'Original')
    plot_foundation(optimized_params, 'green', 'lightgreen', 'Optimized')

    ax.set_aspect('equal')
    plt.xlabel('Width (m)')
    plt.ylabel('Height (m)')
    plt.legend()
    plt.title('Foundation Comparison')
    return fig

def run_calculations(F_z, F_RES, M_RES, rho_conc, rho_ballast_wet, rho_water, params):
    total_weight, C1, C2, C3, C4 = calculate_foundation_weight(params, rho_conc)
    p_min, p_max, B_wet, W, net_load = calculate_pressures(params, F_z, F_RES, M_RES, rho_conc, rho_ballast_wet, rho_water)[:5]

    result = (f"Calculated foundation dimensions and weights:\n"
              f"d1 = {params[0]:.2f} m\n"
              f"d2 = {params[1]:.2f} m\n"
              f"h1 = {params[2]:.2f} m\n"
              f"h2 = {params[3]:.2f} m\n"
              f"h3 = {params[4]:.2f} m\n"
              f"h4 = {params[5]:.2f} m\n"
              f"h5 = {params[6]:.2f} m\n"
              f"b1 = {params[7]:.2f} m\n"
              f"b2 = {params[8]:.2f} m\n"
              f"\nTotal weight: {total_weight:.2f} kN\n"
              f"p_min: {p_min:.2f} kN/m²\n"
              f"p_max: {p_max:.2f} kN/m²\n"
              f"B_wet: {B_wet:.2f} kN\n"
              f"W: {W:.2f} kN\n"
              f"F_z: {F_z:.2f} kN\n"
              f"net_load: {net_load:.2f} kN")

    return result

def optimize_foundation(F_z, F_RES, M_RES, rho_conc, rho_ballast_wet, rho_water, initial_params, h_anchor):
    bounds = [(5, 30), (5, 30), (0.3, 4), (0.3, 4), (0.3, 4), (0.3, 4), (0.3, 4), (5, 30), (5, 30)]

    def objective(x):
        params = [x[0], initial_params[1], x[1], x[2], x[3], initial_params[5], initial_params[6], initial_params[7], initial_params[8]]
        _, _, _, _, _, total_weight = calculate_pressures(params, F_z, F_RES, M_RES, rho_conc, rho_ballast_wet, rho_water)
        return total_weight

    def constraint_pmin(x):
        params = [x[0], initial_params[1], x[1], x[2], x[3], initial_params[5], initial_params[6], initial_params[7], initial_params[8]]
        p_min, _, _, _, _, _ = calculate_pressures(params, F_z, F_RES, M_RES, rho_conc, rho_ballast_wet, rho_water)
        return p_min - 0.5  # Ensure p_min is greater than 0.5 kN/m²

    def constraint_theta(x):
        params = [x[0], initial_params[1], x[1], x[2], x[3], initial_params[5], initial_params[6], initial_params[7], initial_params[8]]
        d1, d2, h2 = params[0], params[1], params[3]
        theta = np.degrees(np.arctan(h2 / ((d1 - d2) / 2)))
        return 13 - theta

    def constraint_h3(x):
        params = [x[0], initial_params[1], x[1], x[2], x[3], initial_params[5], initial_params[6], initial_params[7], initial_params[8]]
        h3, h1, h2 = params[4], params[2], params[3]
        return (h1 + h2) - h3

    def constraint_anchor(x):
        params = [x[0], initial_params[1], x[1], x[2], x[3], initial_params[5], initial_params[6], initial_params[7], initial_params[8]]
        h1, h2, h3, h4, h5 = params[2], params[3], params[4], params[5], params[6]
        return (h1 + h2 + h3 + h4 + h5) - (h_anchor + 0.25)

    cons = [{'type': 'ineq', 'fun': constraint_pmin},
            {'type': 'ineq', 'fun': constraint_theta},
            {'type': 'ineq', 'fun': constraint_h3},
            {'type': 'ineq', 'fun': constraint_anchor}]

    try:
        result = minimize(objective, [initial_params[0], initial_params[2], initial_params[3], initial_params[4]],
                          bounds=[bounds[0], bounds[2], bounds[3], bounds[4]], constraints=cons, method='trust-constr')

        if result.success:
            optimized_params = result.x
            params = [optimized_params[0], initial_params[1], optimized_params[1], optimized_params[2], optimized_params[3], initial_params[5], initial_params[6], initial_params[7], initial_params[8]]
            total_weight, C1, C2, C3, C4 = calculate_foundation_weight(params, rho_conc)
            p_min, p_max, B_wet, W, net_load = calculate_pressures(params, F_z, F_RES, M_RES, rho_conc, rho_ballast_wet, rho_water)[:5]

            result_output = (f"Optimized foundation dimensions and weights:\n"
                             f"d1 = {params[0]:.2f} m\n"
                             f"d2 = {params[1]:.2f} m\n"
                             f"h1 = {params[2]:.2f} m\n"
                             f"h2 = {params[3]:.2f} m\n"
                             f"h3 = {params[4]:.2f} m\n"
                             f"h4 = {params[5]:.2f} m\n"
                             f"h5 = {params[6]:.2f} m\n"
                             f"b1 = {params[7]:.2f} m\n"
                             f"b2 = {params[8]:.2f} m\n"
                             f"h_anchor = {h_anchor:.2f} m\n"
                             f"\nTotal weight: {total_weight:.2f} kN\n"
                             f"p_min: {p_min:.2f} kN/m²\n"
                             f"p_max: {p_max:.2f} kN/m²\n"
                             f"B_wet: { B_wet:.2f} kN\n"
                             f"W: {W:.2f} kN\n"
                             f"F_z: {F_z:.2f} kN\n"
                             f"net_load: {net_load:.2f} kN")

            fig = plot_foundation_comparison(initial_params, params)
            return result_output, fig
        else:
            return f"Optimization failed: {result.message}", None
    except Exception as e:
        return f"Optimization failed due to an exception: {e}", None

# Streamlit Interface
st.title("Foundation Optimization")

st.sidebar.header("Input Parameters")
F_z = st.sidebar.number_input('F_z (kN)', value=3300.0)
F_RES = st.sidebar.number_input('F_RES (kN)', value=511.9)
M_z = st.sidebar.number_input('M_z (kNm)', value=2264.2)
M_RES = st.sidebar.number_input('M_RES (kNm)', value=39122.08)
q_max = st.sidebar.number_input('q_max (kPa)', value=200.0)
rho_conc = st.sidebar.number_input('ρ_conc (kN/m³)', value=24.5)
rho_ballast_wet = st.sidebar.number_input('ρ_ballast_wet (kN/m³)', value=20.0)
rho_ballast_dry = st.sidebar.number_input('ρ_ballast_dry (kN/m³)', value=18.0)
d1 = st.sidebar.number_input('d1 (m)', value=21.6)
d2 = st.sidebar.number_input('d2 (m)', value=6.0)
h1 = st.sidebar.number_input('h1 (m)', value=0.5)
h2 = st.sidebar.number_input('h2 (m)', value=1.4)
h3 = st.sidebar.number_input('h3 (m)', value=0.795)
h4 = st.sidebar.number_input('h4 (m)', value=0.1)
h5 = st.sidebar.number_input('h5 (m)', value=0.25)
b1 = st.sidebar.number_input('b1 (m)', value=6.0)
b2 = st.sidebar.number_input('b2 (m)', value=5.5)
h_anchor = st.sidebar.number_input('h_anchor (m)', value=2.7)

initial_params = [d1, d2, h1, h2, h3, h4, h5, b1, b2]

st.header("Run Calculations")
if st.button("Run Calculations"):
    result_output = run_calculations(F_z, F_RES, M_RES, rho_conc, rho_ballast_wet, -9.81, initial_params)
    st.text(result_output)

st.header("Optimize Foundation")
if st.button("Optimize Foundation"):
    result_output, fig = optimize_foundation(F_z, F_RES, M_RES, rho_conc, rho_ballast_wet, -9.81, initial_params, h_anchor)
    st.text(result_output)
    if fig is not None:
        st.pyplot(fig)
