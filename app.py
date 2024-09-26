import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from bending_moment_calculator import BendingMomentParams, calculate_bending_moment  # New Import

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
    p_min, p_max, B_wet, B_dry, W, net_load, vertical_load = calculate_pressures(params, F_z, F_RES, M_RES, rho_conc, rho_ballast_wet, rho_water, rho_ballast_dry)[:7]

    result = {
        "Parameter": ["d1", "d2", "h1", "h2", "h3", "h4", "h5", "b1", "b2", "p_min", "p_max"],
        "Value": [f"{val:.3f} m" for val in params] + [f"{p_min:.3f} kN/m²", f"{p_max:.3f} kN/m²"]
    }

    concrete_volume = sum([C1, C2, C3, C4])
    return result, concrete_volume, B_dry

def calculate_costs(concrete_volume, steel_weight, ballast_weight, cost_concrete=250, cost_steel=785, cost_ballast=20):
    concrete_cost = concrete_volume * cost_concrete
    steel_cost = steel_weight * cost_steel
    ballast_cost = ballast_weight * (cost_ballast / 10)  # Assuming ballast_weight is in tons
    total_cost = concrete_cost + steel_cost + ballast_cost
    return concrete_cost, steel_cost, ballast_cost, total_cost

def optimize_foundation(F_z, F_RES, M_RES, rho_conc, rho_ballast_wet, rho_water, rho_ballast_dry, initial_params, h_anchor):
    bounds = [(5, 30), (5, 30), (0.3, 4), (0.3, 4), (0.3, 4), (0.3, 4), (0.3, 4), (5, 30), (5, 30)]

    def objective(x):
        # Extract parameters
        d1, d2, h1, h2, h3, h4, h5, b1, b2 = x
        params = [d1, d2, h1, h2, h3, h4, h5, b1, b2]
        
        # Calculate concrete volume
        total_weight, C1, C2, C3, C4 = calculate_weights(params, rho_conc)
        concrete_volume = C1 + C2 + C3 + C4
        
        # Calculate ballast and buoyancy
        B_wet, B_dry, W = calculate_ballast_and_buoyancy(params, C2, C4, rho_ballast_wet, rho_water, rho_ballast_dry)
        
        # Instantiate BendingMomentParams
        bending_moment_params = BendingMomentParams(
            Fz_ULS=F_z,
            load_factor_gamma_f=1.2,          # Replace with actual value if dynamic
            MRes_without_Vd=M_RES,
            safety_factor_favorable=0.9,      # Replace with actual value if dynamic
            d1=d1,
            d2=d2,
            h1=h1,
            h2=h2,
            h3=h3,
            h4=h4,
            h5=h5,
            b1=b1,
            b2=b2,
            rho_conc=rho_conc,
            rho_ballast_wet=rho_ballast_wet
        )
        
        # Calculate bending moment Mt
        Mt = calculate_bending_moment(bending_moment_params, total_weight, B_wet)
        
        # Define weights for objective components
        alpha = 1.0  # Weight for concrete volume
        beta = 1.0   # Weight for Mt
        
        if Mt is None:
            return 1e6  # High penalty if Mt cannot be calculated
        
        return alpha * concrete_volume + beta * Mt  # Weighted sum objective

    # Define constraints if any (Placeholder: Modify as per actual constraints)
    def constraint_pmin(x):
        params = [x[0], initial_params[1], x[2], x[3], x[4], initial_params[5], x[6], initial_params[7], initial_params[8]]
        p_min, _, _, _, _, _, _ = calculate_pressures(params, F_z, F_RES, M_RES, rho_conc, rho_ballast_wet, rho_water, rho_ballast_dry)[:7]
        return p_min - 100  # Example: p_min should be greater than 100 kN/m²

    def constraint_theta(x):
        params = [x[0], initial_params[1], x[2], x[3], x[4], initial_params[5], x[6], initial_params[7], initial_params[8]]
        d1, d2, h2 = params[0], params[1], params[3]
        theta = np.degrees(np.arctan(h2 / ((d1 - d2) / 2)))
        return 12 - theta  # Example: theta should be less than 12 degrees

    def constraint_h3(x):
        params = [x[0], initial_params[1], x[2], x[3], x[4], initial_params[5], x[6], initial_params[7], initial_params[8]]
        h3, h1, h2 = params[4], params[2], params[3]
        return (h1 + h2) - h3  # Example constraint

    def constraint_anchor(x):
        params = [x[0], initial_params[1], x[2], x[3], x[4], initial_params[5], x[6], initial_params[7], initial_params[8]]
        h1, h2, h3, h4, h5 = params[2], params[3], params[4], params[5], params[6]
        return (h1 + h2 + h3 + h4 + h5) - (h_anchor + 0.25)  # Example constraint

    def constraint_h1_h2_ratio(x):
        params = [x[0], initial_params[1], x[2], x[3], x[4], initial_params[5], x[6], initial_params[7], initial_params[8]]
        h1, h2, h3 = params[2], params[3], params[4]
        return h1 + h2 - 0.6 * (h1 + h2 + h3)  # Example constraint

    cons = [
        {'type': 'ineq', 'fun': constraint_pmin},
        {'type': 'ineq', 'fun': constraint_theta},
        {'type': 'ineq', 'fun': constraint_h3},
        {'type': 'ineq', 'fun': constraint_anchor},
        {'type': 'ineq', 'fun': constraint_h1_h2_ratio}
    ]

    # Initial guess
    x0 = initial_params

    # Perform optimization
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)

    if result.success:
        optimized_params = result.x
        d1_opt, d2_opt, h1_opt, h2_opt, h3_opt, h4_opt, h5_opt, b1_opt, b2_opt = optimized_params
        params_opt = [d1_opt, d2_opt, h1_opt, h2_opt, h3_opt, h4_opt, h5_opt, b1_opt, b2_opt]
        total_weight_opt, C1_opt, C2_opt, C3_opt, C4_opt = calculate_weights(params_opt, rho_conc)
        concrete_volume_opt = C1_opt + C2_opt + C3_opt + C4_opt
        p_min_opt, p_max_opt, B_wet_opt, B_dry_optimal, W_opt, net_load_opt, vertical_load_opt = calculate_pressures(params_opt, F_z, F_RES, M_RES, rho_conc, rho_ballast_wet, rho_water, rho_ballast_dry)[:7]

        # Calculate bending moment Mt using the bending moment calculator
        bending_moment_params_opt = BendingMomentParams(
            Fz_ULS=F_z,
            load_factor_gamma_f=1.2,          # Replace with actual value if dynamic
            MRes_without_Vd=M_RES,
            safety_factor_favorable=0.9,      # Replace with actual value if dynamic
            d1=d1_opt,
            d2=d2_opt,
            h1=h1_opt,
            h2=h2_opt,
            h3=h3_opt,
            h4=h4_opt,
            h5=h5_opt,
            b1=b1_opt,
            b2=b2_opt,
            rho_conc=rho_conc,
            rho_ballast_wet=rho_ballast_wet
        )

        Mt_opt = calculate_bending_moment(bending_moment_params_opt, total_weight_opt, B_wet_opt)

        # If Mt cannot be calculated, assign a high penalty or handle accordingly
        if Mt_opt is None:
            Mt_opt = 1e6  # Example penalty value

        # Calculate steel and ballast weights
        optimized_steel = 0.135 * concrete_volume_opt
        optimized_ballast = B_dry_optimal  # Assuming B_dry represents ballast weight

        # Calculate costs
        optimized_concrete_cost, optimized_steel_cost, optimized_ballast_cost, optimized_total_cost = calculate_costs(
            optimized_concrete_volume, optimized_steel, optimized_ballast
        )

        # Prepare result output
        result_output = {
            "Parameter": ["d1 (m)", "d2 (m)", "h1 (m)", "h2 (m)", "h3 (m)", "h4 (m)", "h5 (m)", "b1 (m)", "b2 (m)", "p_min (kN/m²)", "p_max (kN/m²)", "Concrete Volume (m³)", "Mt (kNm)"],
            "Value": [
                f"{d1_opt:.3f}", f"{d2_opt:.3f}", f"{h1_opt:.3f}", f"{h2_opt:.3f}", 
                f"{h3_opt:.3f}", f"{h4_opt:.3f}", f"{h5_opt:.3f}", f"{b1_opt:.3f}", 
                f"{b2_opt:.3f}", f"{p_min_opt:.3f}", f"{p_max_opt:.3f}", 
                f"{concrete_volume_opt:.3f}", f"{Mt_opt:.2f}"
            ]
        }

        # Plot foundation comparison
        fig = plot_foundation_comparison(initial_params, params_opt)

        return result_output, concrete_volume_opt, B_dry_optimal, fig
    else:
        result_output = {
            "Parameter": ["Error"],
            "Value": [f"Optimization failed: {result.message}"]
        }
        return result_output, None, None, None

def plot_cost_comparison(original_cost, optimized_cost, original_breakdown, optimized_breakdown):
    """Plot cost comparison between original and optimized designs."""
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

    # Add detailed cost breakdown as text below the plot
    detailed_text = (
        f"Original Breakdown:\n"
        f"Concrete Cost: £{original_breakdown[0]:,.2f}\n"
        f"Steel Cost: £{original_breakdown[1]:,.2f}\n"
        f"Ballast Cost: £{original_breakdown[2]:,.2f}\n\n"
        f"Optimized Breakdown:\n"
        f"Concrete Cost: £{optimized_breakdown[0]:,.2f}\n"
        f"Steel Cost: £{optimized_breakdown[1]:,.2f}\n"
        f"Ballast Cost: £{optimized_breakdown[2]:,.2f}"
    )
    plt.figtext(0.5, -0.2, detailed_text, ha='center', va='top', fontsize=10)
    plt.subplots_adjust(bottom=0.4)

    return fig

def plot_concrete_volume(volume_data):
    """Plot concrete volume comparison."""
    fig, ax = plt.subplots()
    bars = ax.barh(volume_data['Volume'], volume_data['Concrete Volume (m³)'], color=['red', 'green'])
    for bar, label in zip(bars, [f"{v:.3f} m³" for v in volume_data['Concrete Volume (m³)']]):
        width = bar.get_width()
        ax.text(width / 2, bar.get_y() + bar.get_height() / 2, label, ha='center', va='center', color='black')
    plt.xlabel('Concrete Volume (m³)')
    plt.title('Concrete Volume Comparison')
    return fig

def plot_steel_and_ballast(data):
    """Plot steel and ballast weight comparison."""
    fig, ax = plt.subplots()
    bars = ax.bar(data['Category'], data['Weight (t)'], color=['blue', 'blue', 'orange', 'orange'])
    for bar, label in zip(bars, [f"{v:.3f} t" for v in data['Weight (t)']]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height / 2, label, ha='center', va='center', color='black')
    plt.xlabel('Category')
    plt.ylabel('Weight (t)')
    plt.title('Steel and Ballast Weight Comparison')
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
rho_ballast_wet = st.sidebar.number_input(r'$\rho_{ballast\,wet}$ (kN/m³)', value=20.000, format="%.3f')
rho_ballast_dry = st.sidebar.number_input(r'$\rho_{ballast\,dry}$ (kN/m³)', value=18.000, format="%.3f')

st.sidebar.subheader("Dimensions")
d1 = st.sidebar.number_input('d1 (m)', value=21.600, format="%.3f")
d2 = st.sidebar.number_input('d2 (m)', value=6.000, format="%.3f")
h1 = st.sidebar.number_input('h1 (m)', value=0.500, format="%.3f")
h2 = st.sidebar.number_input('h2 (m)', value=1.400, format="%.3f")
h3 = st.sidebar.number_input('h3 (m)', value=0.795, format="%.3f")
h4 = st.sidebar.number_input('h4 (m)', value=0.100, format="%.3f")
h5 = st.sidebar.number_input('h5 (m)', value=0.250, format="%.3f")
b1 = st.sidebar.number_input('b1 (m)', value=6.000, format="%.3f")
b2 = st.sidebar.number_input('b2 (m)', value=5.500, format="%.3f')
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
    st.markdown(result_df.to_html(index=False), unsafe_allow_html=True)
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
        st.markdown(result_df.to_html(index=False), unsafe_allow_html=True)

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
        original_steel = 0.135 * st.session_state['original_concrete_volume'] if st.session_state['original_concrete_volume'] else 0
        optimized_steel = 0.135 * optimized_concrete_volume

        optimized_ballast = B_dry_optimal  # Assuming B_dry represents ballast weight

        weight_data = pd.DataFrame({
            'Category': ['Optimized Steel', 'Optimized Ballast'],
            'Weight (t)': [optimized_steel, optimized_ballast]
        })

        fig_weight = plot_steel_and_ballast(weight_data)
        st.pyplot(fig_weight)

        # Calculate costs
        optimized_concrete_cost, optimized_steel_cost, optimized_ballast_cost, optimized_total_cost = calculate_costs(
            optimized_concrete_volume, optimized_steel, optimized_ballast
        )

        # Display costs
        st.write("Optimized Costs:")
        st.write(f"Concrete: £{optimized_concrete_cost:,.2f}")
        st.write(f"Steel: £{optimized_steel_cost:,.2f}")
        st.write(f"Ballast: £{optimized_ballast_cost:,.2f}")
        st.write(f"Total: £{optimized_total_cost:,.2f}")

        # Plot cost comparison
        original_concrete_cost = 0  # Placeholder: Replace with actual original costs if available
        original_steel_cost = 0      # Placeholder
        original_ballast_cost = 0    # Placeholder
        original_total_cost = 0      # Placeholder

        optimized_breakdown = (optimized_concrete_cost, optimized_steel_cost, optimized_ballast_cost)
        original_breakdown = (original_concrete_cost, original_steel_cost, original_ballast_cost)

        fig_cost = plot_cost_comparison(
            original_total_cost, optimized_total_cost,
            original_breakdown, optimized_breakdown
        )
        st.pyplot(fig_cost)

    else:
        st.error(f"Optimization failed: {result_output['Value'][0]}")
