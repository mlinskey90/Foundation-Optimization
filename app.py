import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize
import pandas as pd

# Set the default font to Helvetica for matplotlib
plt.rcParams['font.sans-serif'] = ['Helvetica']
plt.rcParams['font.family'] = 'sans-serif'

# Define the necessary functions
def calculate_foundation_weight(params, rho_conc):
    d1, d2, h1, h2, h3, h4, h5, b1, b2 = params
    C1 = (np.pi * d1**2 / 4) * h1
    C2 = (1/3) * np.pi * ((d1/2)**2 + (d1/2 * d2/2) + (d2/2)**2) * h2
    C3 = (np.pi * d2**2 / 4) * h3
    C4 = (1/3) * np.pi * ((b1/2)**2 + (b1/2 * b2/2) + (b2/2)**2) * h5
    total_weight = (C1 + C2 + C3 + C4) * rho_conc

    return total_weight, C1, C2, C3, C4

def calculate_ballast_and_buoyancy(params, C2, C4, rho_ballast_wet, rho_water):
    d1, d2, h1, h2, h3, h4 = params[0], params[1], params[2], params[3], params[4], params[5]
    h_water = h1 + h2 + h3 - h4
    B_wet = ((np.pi * d1**2 / 4) * (h2 + h3 - h4) - C2 - (np.pi * d2**2 / 4) * (h3 - h4)) * rho_ballast_wet
    W = (((np.pi * (d1 ** 2)) / 4) * h_water + C4) * rho_water
    return B_wet, W

def net_vertical_load(params, F_z, rho_conc, rho_ballast_wet, rho_water):
    total_weight, C1, C2, C3, C4 = calculate_foundation_weight(params, rho_conc)
    B_wet, W = calculate_ballast_and_buoyancy(params, C2, C4, rho_ballast_wet, rho_water)
    net_load = W + B_wet + total_weight + F_z
    return net_load, total_weight, B_wet, W

def calculate_pressures(params, F_z, F_RES, M_RES, rho_conc, rho_ballast_wet, rho_water):
    d1 = params[0]
    vertical_load, total_weight, B_wet, W = net_vertical_load(params, F_z, rho_conc, rho_ballast_wet, rho_water)
    M_RES2 = M_RES + F_RES * (params[2] + params[3] + params[4])  # M_RES2 = M_RES + F_RES x (h1 + h2 + h3)
    resultant_moment = M_RES2

    area = np.pi * d1**2 / 4
    p_min = (vertical_load / area) - (resultant_moment / (np.pi * d1**3 / 32))
    p_max = (vertical_load / area) + (resultant_moment / (np.pi * d1**3 / 32))

    return p_min, p_max, B_wet, W, vertical_load, total_weight

def plot_foundation(ax, params, edgecolor, fillcolor, label):
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

def plot_foundation_comparison(original_params, optimized_params):
    fig, ax = plt.subplots(figsize=(20, 15))

    plot_foundation(ax, original_params, 'black', 'grey', 'Original')
    plot_foundation(ax, optimized_params, 'green', 'lightgreen', 'Optimized')

    ax.set_aspect('equal')
    plt.xlabel('Width (m)')
    plt.ylabel('Height (m)')
    plt.legend()
    plt.title('Foundation Comparison')
    return fig

def run_calculations(F_z, F_RES, M_RES, rho_conc, rho_ballast_wet, rho_water, params):
    total_weight, C1, C2, C3, C4 = calculate_foundation_weight(params, rho_conc)
    p_min, p_max, B_wet, W, net_load = calculate_pressures(params, F_z, F_RES, M_RES, rho_conc, rho_ballast_wet, rho_water)[:5]

    result = {
        "Parameter": [
            "d1", "d2", "h1", "h2", "h3", "h4", "h5", "b1", "b2",
            "C1", "C2", "C3", "C4",
            "Total weight", "p_min", "p_max", "B_wet", "W", "F_z", "net_load"
        ],
        "Value": [
            f"{params[0]:.3f} m", f"{params[1]:.3f} m", f"{params[2]:.3f} m", f"{params[3]:.3f} m", f"{params[4]:.3f} m",
            f"{params[5]::.3f} m", f"{params[6]::.3f} m", f"{params[7]::.3f} m", f"{params[8]::.3f} m",
            f"{C1:.3f} m³", f"{C2:.3f} m³", f"{C3:.3f} m³", f"{C4:.3f} m³",
            f"{total_weight:.3f} kN", f"{p_min:.3f} kN/m²", f"{p_max:.3f} kN/m²", f"{B_wet:.3f} kN", f"{W:.3f} kN",
            f"{F_z:.3f} kN", f"{net_load:.3f} kN"
        ]
    }

    concrete_volume = (C1 + C2 + C3 + C4)

    return result, concrete_volume

def optimize_foundation(F_z, F_RES, M_RES, rho_conc, rho_ballast_wet, rho_water, initial_params, h_anchor):
    bounds = [(5, 30), (0.3, 4), (0.3, 4), (0.3, 4)]

    def objective(x):
        params = [x[0], initial_params[1], x[1], x[2], x[3], initial_params[5], initial_params[6], initial_params[7], initial_params[8]]
        return calculate_foundation_weight(params, rho_conc)[0]

    def constraint_pmin(x):
        params = [x[0], initial_params[1], x[1], x[2], x[3], initial_params[5], initial_params[6], initial_params[7], initial_params[8]]
        return calculate_pressures(params, F_z, F_RES, M_RES, rho_conc, rho_ballast_wet, rho_water)[0]

    def constraint_theta(x):
        params = [x[0], initial_params[1], x[1], x[2], x[3], initial_params[5], initial_params[6], initial_params[7], initial_params[8]]
        d1, d2, h2 = params[0], params[1], params[3]
        theta = np.degrees(np.arctan(h2 / ((d1 - d2) / 2)))
        return 13 - theta

    def constraint_h3(x):
        params = [x[0], initial_params[1], x[1], x[2], x[3], initial_params[5], initial_params[6], initial_params[7], initial_params[8]]
        return (params[2] + params[3]) - params[4]

    def constraint_anchor(x):
        params = [x[0], initial_params[1], x[1], x[2], x[3], initial_params[5], initial_params[6], initial_params[7], initial_params[8]]
        return (params[2] + params[3] + params[4] + params[5] + params[6]) - (h_anchor + 0.25)

    cons = [{'type': 'ineq', 'fun': constraint_pmin},
            {'type': 'ineq', 'fun': constraint_theta},
            {'type': 'ineq', 'fun': constraint_h3},
            {'type': 'ineq', 'fun': constraint_anchor}]

    try:
        result = minimize(objective, [initial_params[0], initial_params[2], initial_params[3], initial_params[4]],
                          bounds=bounds, constraints=cons, method='trust-constr')

        if result.success:
            optimized_params = result.x
            params = [optimized_params[0], initial_params[1], optimized_params[1], optimized_params[2], optimized_params[3], initial_params[5], initial_params[6], initial_params[7], initial_params[8]]
            total_weight, C1, C2, C3, C4 = calculate_foundation_weight(params, rho_conc)
            p_min, p_max, B_wet, W, net_load = calculate_pressures(params, F_z, F_RES, M_RES, rho_conc, rho_ballast_wet, rho_water)[:5]

            result_output = {
                "Parameter": [
                    "d1", "d2", "h1", "h2", "h3", "h4", "h5", "b1", "b2",
                    "Total weight", "p_min", "p_max", "B_wet", "W", "F_z", "net_load"
                ],
                "Value": [
                    f"{params[0]:.3f} m", f"{params[1]:.3f} m", f"{params[2]:.3f} m", f"{params[3]:.3f} m", f"{params[4]:.3f} m",
                    f"{params[5]:.3f} m", f"{params[6]:.3f} m", f"{params[7]:.3f} m", f"{params[8]:.3f} m",
                    f"{total_weight:.3f} kN", f"{p_min:.3f} kN/m²", f"{p_max:.3f} kN/m²", f"{B_wet:.3f} kN", f"{W:.3f} kN",
                    f"{F_z:.3f} kN", f"{net_load:.3f} kN"
                ]
            }

            optimized_concrete_volume = (C1 + C2 + C3 + C4)
            fig = plot_foundation_comparison(initial_params, params)
            return result_output, optimized_concrete_volume, fig
        else:
            return {"Parameter": [], "Value": [f"Optimization failed: {result.message}"]}, None, None
    except Exception as e:
        return {"Parameter": [], "Value": [f"Optimization failed due to an exception: {e}"]}, None, None

# Streamlit Interface
st.title("Foundation Optimization")

# Load and display the image above the "Run Calculations" button
image_path = "foundation.PNG"
st.image(image_path, caption="Foundation Diagram", use_column_width=True)

st.sidebar.header("Input Parameters")

# Load Cases
st.sidebar.subheader("Load Cases")
F_z = st.sidebar.number_input(r'$F_z$ (kN)', value=3300.000, format="%.3f")
F_RES = st.sidebar.number_input(r'$F_{RES}$ (kN)', value=511.900, format="%.3f")
M_z = st.sidebar.number_input(r'$M_z$ (kNm)', value=2264.200, format="%.3f")
M_RES = st.sidebar.number_input(r'$M_{RES}$ (kNm)', value=39122.080, format="%.3f")

# Material Properties
st.sidebar.subheader("Material Properties")
q_max = st.sidebar.number_input(r'$q_{max}$ (kPa)', value=200.000, format="%.3f")
rho_conc = st.sidebar.number_input(r'$\rho_{conc}$ (kN/m³)', value=24.500, format="%.3f")
rho_ballast_wet = st.sidebar.number_input(r'$\rho_{ballast\,wet}$ (kN/m³)', value=20.000, format="%.3f")
rho_ballast_dry = st.sidebar.number_input(r'$\rho_{ballast\,dry}$ (kN/m³)', value=18.000, format="%.3f")

# Dimensions
st.sidebar.subheader("Dimensions")
d1 = st.sidebar.number_input('d1 (m)', value=21.600, format="%.3f")
d2 = st.sidebar.number_input('d2 (m)', value=6.000, format="%.3f")
h1 = st.sidebar.number_input('h1 (m)', value=0.500, format="%.3f")
h2 = st.sidebar.number_input('h2 (m)', value=1.400, format="%.3f")
h3 = st.sidebar.number_input('h3 (m)', value=0.795, format="%.3f")
h4 = st.sidebar.number_input('h4 (m)', value=0.100, format="%.3f")
h5 = st.sidebar.number_input('h5 (m)', value=0.250, format="%.3f")
b1 = st.sidebar.number_input('b1 (m)', value=6.000, format="%.3f")
b2 = st.sidebar.number_input('b2 (m)', value=5.500, format="%.3f")
h_anchor = st.sidebar.number_input(r'$h_{anchor}$ (m)', value=2.700, format="%.3f")

initial_params = [d1, d2, h1, h2, h3, h4, h5, b1, b2]

# Initialize session state for original concrete volume
if 'original_concrete_volume' not in st.session_state:
    st.session_state['original_concrete_volume'] = None

st.header("Run Calculations")
if st.button("Run Calculations"):
    result_output, original_concrete_volume = run_calculations(F_z, F_RES, M_RES, rho_conc, rho_ballast_wet, -9.81, initial_params)
    st.session_state['original_concrete_volume'] = original_concrete_volume  # Store the original concrete volume in session state
    result_df = pd.DataFrame(result_output)
    st.dataframe(result_df.style.hide(axis="index"), use_container_width=True)
    st.subheader("Concrete Volume")
    st.write(f"Original Concrete Volume: {original_concrete_volume:.3f} m³")

st.header("Optimize Foundation")
if st.button("Optimize Foundation"):
    result_output, optimized_concrete_volume, fig = optimize_foundation(F_z, F_RES, M_RES, rho_conc, rho_ballast_wet, -9.81, initial_params, h_anchor)
    result_df = pd.DataFrame(result_output)
    st.dataframe(result_df.style.hide(axis="index"), use_container_width=True)
    if fig is not None:
        st.pyplot(fig)
        st.subheader("Concrete Volume Comparison")
        if st.session_state['original_concrete_volume'] is not None:
            st.write(f"Original Concrete Volume: {st.session_state['original_concrete_volume']:.3f} m³")
        st.write(f"Optimized Concrete Volume: {optimized_concrete_volume:.3f} m³")
        if st.session_state['original_concrete_volume'] is not None:
            volume_data = pd.DataFrame({
                'Volume': ['Original', 'Optimized'],
                'Concrete Volume (m³)': [st.session_state['original_concrete_volume'], optimized_concrete_volume]
            })

            def plot_concrete_volume(volume_data):
                fig, ax = plt.subplots()
                bars = ax.barh(volume_data['Volume'], volume_data['Concrete Volume (m³)'], color=['red', 'green'])

                for bar, label in zip(bars, [f"{v:.3f} m³" for v in volume_data['Concrete Volume (m³)']]):
                    width = bar.get_width()
                    ax.text(width / 2, bar.get_y() + bar.get_height() / 2, label, ha='center', va='center', color='black')

                plt.xlabel('Concrete Volume (m³)')
                plt.title('Concrete Volume Comparison')
                return fig

            fig = plot_concrete_volume(volume_data)
            st.pyplot(fig)
