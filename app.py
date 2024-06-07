import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Function definitions with docstrings
def calculate_volumes(params):
    """Calculate volumes of different parts of the foundation."""
    d1, d2, h1, h2, h3, h4, h5, b1, b2 = params
    C1 = (np.pi * d1**2 / 4) * h1
    C2 = (1/3) * np.pi * ((d1/2)**2 + (d1/2 * d2/2) + (d2/2)**2) * h2
    C3 = (np.pi * d2**2 / 4) * h3
    C4 = (1/3) * np.pi * ((b1/2)**2 + (b1/2 * b2/2) + (b2/2)**2) * h5
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

def plot_foundation_comparison(original_params, optimized_params):
    """Plot comparison between original and optimized foundation designs."""
    fig, ax = plt.subplots(figsize=(20, 15))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.title.set_color('black')
    
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
    plt.xlabel('Width (m)', color='black')
    plt.ylabel('Height (m)', color='black')
    plt.legend()
    plt.title('Foundation Comparison', color='black')
    return fig

def run_calculations(F_z, F_RES, M_RES, rho_conc, rho_ballast_wet, rho_water, rho_ballast_dry, params):
    total_weight, C1, C2, C3, C4 = calculate_weights(params, rho_conc)
    p_min, p_max, B_wet, B_dry, W, net_load = calculate_pressures(params, F_z, F_RES, M_RES, rho_conc, rho_ballast_wet, rho_water, rho_ballast_dry)[:6]

    result = {
        "Parameter": ["d1", "d2", "h1", "h2", "h3", "h4", "h5", "b1", "b2", "p_min", "p_max"],
        "Value": [f"{val:.3f} m" for val in params] + [f"{p_min:.3f} kN/m²", f"{p_max:.3f} kN/m²"]
    }

    concrete_volume = sum([C1, C2, C3, C4])
    return result, concrete_volume, B_dry

def calculate_costs(concrete_volume, steel_weight, ballast_weight, cost_concrete=120, cost_steel=600, cost_ballast=15):
    concrete_cost = concrete_volume * cost_concrete
    steel_cost = steel_weight * cost_steel
    ballast_cost = ballast_weight * cost_ballast
    total_cost = concrete_cost + steel_cost + ballast_cost
    return concrete_cost, steel_cost, ballast_cost, total_cost

def optimize_foundation(F_z, F_RES, M_RES, rho_conc, rho_ballast_wet, rho_water, rho_ballast_dry, initial_params, h_anchor):
    bounds = [(5, 30), (5, 30), (0.3, 4), (0.3, 4), (0.3, 4), (0.3, 4), (0.3, 4), (5, 30), (5, 30)]

    def objective(x):
        params = [x[0], initial_params[1], x[1], x[2], x[3], initial_params[5], initial_params[6], initial_params[7], initial_params[8]]
        return calculate_weights(params, rho_conc)[0]

    def constraint_pmin(x):
        params = [x[0], initial_params[1], x[1], x[2], x[3], initial_params[5], initial_params[6], initial_params[7], initial_params[8]]
        return calculate_pressures(params, F_z, F_RES, M_RES, rho_conc, rho_ballast_wet, rho_water, rho_ballast_dry)[0]

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
    
    def constraint_h1_h2_ratio(x):
        params = [x[0], initial_params[1], x[1], x[2], x[3], initial_params[5], initial_params[6], initial_params[7], initial_params[8]]
        h1, h2, h3 = params[2], params[3], params[4]
        return h1 + h2 - 0.6 * (h1 + h2 + h3)

    cons = [{'type': 'ineq', 'fun': constraint_pmin},
            {'type': 'ineq', 'fun': constraint_theta},
            {'type': 'ineq', 'fun': constraint_h3},
            {'type': 'ineq', 'fun': constraint_anchor},
            {'type': 'ineq', 'fun': constraint_h1_h2_ratio}]

    result = minimize(objective, [initial_params[0], initial_params[2], initial_params[3], initial_params[4]],
                      bounds=[bounds[0], bounds[2], bounds[3], bounds[4]], constraints=cons, method='trust-constr')

    if result.success:
        optimized_params = result.x
        params = [optimized_params[0], initial_params[1], optimized_params[1], optimized_params[2], optimized_params[3], initial_params[5], initial_params[6], initial_params[7], initial_params[8]]
        total_weight, C1, C2, C3, C4 = calculate_weights(params, rho_conc)
        p_min, p_max, B_wet, B_dry_optimal, W, net_load = calculate_pressures(params, F_z, F_RES, M_RES, rho_conc, rho_ballast_wet, rho_water, rho_ballast_dry)[:6]

        result_output = {
            "Parameter": ["d1", "d2", "h1", "h2", "h3", "h4", "h5", "b1", "b2", "p_min", "p_max"],
            "Value": [f"{val:.3f} m" for val in params] + [f"{p_min:.3f} kN/m²", f"{p_max:.3f} kN/m²"]
        }

        optimized_concrete_volume = sum([C1, C2, C3, C4])
        fig = plot_foundation_comparison(initial_params, params)
        return result_output, optimized_concrete_volume, B_dry_optimal, fig
    else:
        result_output = {
            "Parameter": ["Error"],
            "Value": [f"Optimization failed: {result.message}"]
        }
        return result_output, None, None, None

def plot_cost_comparison(original_cost, optimized_cost):
    fig, ax = plt.subplots(figsize=(10, 7))
    categories = ['Original', 'Optimized']
    total_costs = [original_cost, optimized_cost]

    bar_width = 0.5
    r1 = np.arange(len(categories))

    bars = ax.bar(r1, total_costs, color='purple', width=bar_width)

    for i in range(len(bars)):
        ax.text(bars[i].get_x() + bars[i].get_width() / 2, bars[i].get_height(), f"£{total_costs[i]:,.2f}", ha='center', va='bottom', color='white')

    ax.set_xlabel('Category', fontweight='bold')
    ax.set_xticks(r1)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Cost (£)')
    ax.set_title('Foundation Cost Comparison')
    return fig

# Streamlit Interface
st.title("Foundation Optimization")

# Load and display the uploaded image
image_path = "foundation.PNG"
st.image(image_path, caption="Foundation Diagram", use_column_width=True)

# Sidebar Inputs
st.sidebar.header("Input Parameters")
st.sidebar.subheader("Load Cases")
F_z = st.sidebar.number_input(r'$F_z$ (kN)', value=3300.000, format="%.3f")
F_RES = st.sidebar.number_input(r'$F_{RES}$ (kN)', value=511.900, format="%.3f")
M_z = st.sidebar.number_input(r'$M_z$ (kNm)', value=2264.200, format="%.3f")
M_RES = st.sidebar.number_input(r'$M_{RES}$ (kNm)', value=39122.080, format="%.3f")

st.sidebar.subheader("Material Properties")
q_max = st.sidebar.number_input(r'$q_{max}$ (kPa)', value=200.000, format="%.3f")
rho_conc = st.sidebar.number_input(r'$\rho_{conc}$ (kN/m³)', value=24.500, format="%.3f")
rho_ballast_wet = st.sidebar.number_input(r'$\rho_{ballast\,wet}$ (kN/m³)', value=20.000, format="%.3f")
rho_ballast_dry = st.sidebar.number_input(r'$\rho_{ballast\,dry}$ (kN/m³)', value=18.000, format="%.3f")

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

# Define rho_water in the code
rho_water = -9.81

# Initialize session state for original concrete volume
if 'original_concrete_volume' not in st.session_state:
    st.session_state['original_concrete_volume'] = None

if 'original_ballast' not in st.session_state:
    st.session_state['original_ballast'] = None

# Placeholder for the boolean flag indicating if the optimization button was clicked
optimize_clicked = st.session_state.get('optimize_clicked', False)

st.header("Run Calculations")
if st.button("Run Calculations", key="run_calculations_button"):
    result_output, original_concrete_volume, B_dry_original = run_calculations(F_z, F_RES, M_RES, rho_conc, rho_ballast_wet, rho_water, rho_ballast_dry, initial_params)
    st.session_state['original_concrete_volume'] = original_concrete_volume
    st.session_state['original_ballast'] = B_dry_original

    result_df = pd.DataFrame(result_output)
    result_html = result_df.to_html(index=False)
    st.markdown(result_html, unsafe_allow_html=True)
    st.subheader("Concrete Volume")
    st.write(f"Original Concrete Volume: {original_concrete_volume:.3f} m³")

st.header("Optimize Foundation")
if st.button("Optimize Foundation", key="optimize_foundation_button"):
    optimize_clicked = True
    st.session_state['optimize_clicked'] = optimize_clicked

if optimize_clicked:
    result_output, optimized_concrete_volume, B_dry_optimal, fig = optimize_foundation(F_z, F_RES, M_RES, rho_conc, rho_ballast_wet, rho_water, rho_ballast_dry, initial_params, h_anchor)

    if result_output["Parameter"][0] != "Error":
        original_values = [f"{val:.3f} m" for val in initial_params]

        result_df = pd.DataFrame(result_output)
        result_df.columns = ["Parameter", "Optimized Value"]
        result_df.insert(1, "Original Value", original_values + ["N/A"] * (len(result_output["Parameter"]) - len(original_values)))

        result_html = result_df.to_html(index=False)
        st.markdown(result_html, unsafe_allow_html=True)

        # Display 2D plot comparison
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
            fig_volume = plot_concrete_volume(volume_data)
            st.pyplot(fig_volume)

        # Additional Calculations for Steel and Ballast
        original_steel = 0.135 * st.session_state['original_concrete_volume']
        optimized_steel = 0.135 * optimized_concrete_volume

        weight_data = pd.DataFrame({
            'Category': ['Original Steel', 'Optimized Steel', 'Original Ballast', 'Optimized Ballast'],
            'Weight (t)': [original_steel, optimized_steel, st.session_state['original_ballast'] * 0.1, B_dry_optimal * 0.1]
        })

        fig_weight = plot_steel_and_ballast(weight_data)
        st.pyplot(fig_weight)

        # Calculate costs
        original_concrete_cost, original_steel_cost, original_ballast_cost, original_total_cost = calculate_costs(
            st.session_state['original_concrete_volume'], original_steel, st.session_state['original_ballast']
        )
        optimized_concrete_cost, optimized_steel_cost, optimized_ballast_cost, optimized_total_cost = calculate_costs(
            optimized_concrete_volume, optimized_steel, B_dry_optimal
        )

        # Plot cost comparison
        fig_cost = plot_cost_comparison(original_total_cost, optimized_total_cost)
        st.pyplot(fig_cost)

    else:
        st.error(f"Optimization failed: {result_output['Value'][0]}")
