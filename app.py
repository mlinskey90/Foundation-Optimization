import streamlit as st
from foundation_geometry import calculate_volumes, calculate_weights, net_vertical_load, calculate_pressures
from bending_moment_calculator import BendingMomentParams, calculate_Mt
import numpy as np
import matplotlib.pyplot as plt

# Streamlit Interface
st.title("Wind Turbine Foundation Optimization")

# Sidebar Inputs
st.sidebar.header("Input Parameters")
st.sidebar.subheader("Load Cases")
F_z = st.sidebar.number_input('F_z (kN)', value=3300.0, format="%.3f")
F_RES = st.sidebar.number_input('F_RES (kN)', value=511.9, format="%.3f")
M_RES = st.sidebar.number_input('M_RES (kNm)', value=39122.08, format="%.3f")

st.sidebar.subheader("Material Properties")
rho_conc = st.sidebar.number_input('Concrete Density (kN/m³)', value=24.5, format="%.3f")
rho_ballast_wet = st.sidebar.number_input('Ballast Density Wet (kN/m³)', value=20.0, format="%.3f")
rho_ballast_dry = st.sidebar.number_input('Ballast Density Dry (kN/m³)', value=18.0, format="%.3f")

st.sidebar.subheader("Dimensions")
d1 = st.sidebar.number_input('d1 (m)', value=21.6, format="%.3f")
d2 = st.sidebar.number_input('d2 (m)', value=6.0, format="%.3f")
h1 = st.sidebar.number_input('h1 (m)', value=0.5, format="%.3f")
h2 = st.sidebar.number_input('h2 (m)', value=1.4, format="%.3f")
h3 = st.sidebar.number_input('h3 (m)', value=0.795, format="%.3f")
h4 = st.sidebar.number_input('h4 (m)', value=0.1, format="%.3f")
h5 = st.sidebar.number_input('h5 (m)', value=0.25, format="%.3f")
b1 = st.sidebar.number_input('b1 (m)', value=6.0, format="%.3f")
b2 = st.sidebar.number_input('b2 (m)', value=5.5, format="%.3f")

params = [d1, d2, h1, h2, h3, h4, h5, b1, b2]

# Run Calculations using imported functions
st.header("Run Calculations")
if st.button("Run Calculations"):
    # Use geometry functions to get initial calculations
    total_weight, C1, C2, C3, C4 = calculate_weights(params, rho_conc)
    net_load, total_weight, B_wet, B_dry, W = net_vertical_load(params, F_z, rho_conc, rho_ballast_wet, -9.81, rho_ballast_dry)
    
    # Create BendingMomentParams instance to use in bending moment calculations
    bending_params = BendingMomentParams(
        Fz_ULS=F_z,
        MRes_without_Vd=M_RES,
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
        rho_ballast_wet=rho_ballast_wet,
        M_z_ULS=0.0,  # Placeholder for z-direction moment
        F_Res_ULS=0.0  # Placeholder for resisting force
    )
    
    # Calculate the bending moment Mt
    Mt = calculate_Mt(bending_params, total_weight, B_wet)

        st.write(f"Calculated Bending Moment (Mt): {Mt:.2f} kNm")

    

    # Optimize dimensions based on Mt

    def calculate_volumes(d1, d2, h1, h2, h3, h4, h5, b1, b2, Mt):

        """

        Adjust foundation dimensions based on bending moment Mt for optimization.

        """

        scaling_factor = 1.0 + (Mt / 1e5)  # Example scaling factor logic

        optimized_d1 = d1 * scaling_factor

        optimized_d2 = d2 * scaling_factor

        return optimized_d1, optimized_d2, scaling_factor



    optimized_d1, optimized_d2, scaling_factor = calculate_volumes(d1, d2, h1, h2, h3, h4, h5, b1, b2, Mt)

    st.write(f"Optimized d1: {optimized_d1:.2f} m, Optimized d2: {optimized_d2:.2f} m (Scaling Factor: {scaling_factor:.2f})")

    

    # Update parameters with optimized values

    params = [optimized_d1, optimized_d2, h1, h2, h3, h4, h5, b1, b2]
    
    # Use pressure calculations
    p_min, p_max, B_wet, B_dry, W, vertical_load, total_weight = calculate_pressures(params, F_z, F_RES, M_RES, rho_conc, rho_ballast_wet, -9.81, rho_ballast_dry)
    
    # Display the results
    if Mt is not None:
        st.write(f"Total Weight: {total_weight:.2f} kN")
        st.write(f"Calculated Bending Moment (Mt): {Mt:.2f} kNm")
        st.write(f"Minimum Pressure (p_min): {p_min:.2f} kN/m²")
        st.write(f"Maximum Pressure (p_max): {p_max:.2f} kN/m²")
    else:
        st.error("Bending moment calculation failed. Please check input parameters.")

# Optional: Plotting the foundation comparison
if st.button("Plot Foundation Comparison"):
    original_params = params
    optimized_params = [optimized_d1, optimized_d2, h1, h2, h3, h4, h5, b1, b2]
    
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
        plinth_x = [-d2 / 2, d2 / 2, d2 / 2, -d2 / 2, -d2 / 2]
        plinth_y = [h1 + h2 + h3, h1 + h2 + h3, h1 + h2, h1 + h2, h1 + h2 + h3]
        haunch_x = [-d1 / 2, d1 / 2, d2 / 2, -d2 / 2, -d1 / 2]
        haunch_y = [h1, h1, h1 + h2, h1 + h2, h1]
        slab_x = [-d1 / 2, d1 / 2, d1 / 2, -d1 / 2, -d1 / 2]
        slab_y = [0, 0, h1, h1, 0]
        downstand_x = [-b1 / 2, -b2 / 2, b2 / 2, b1 / 2, -b1 / 2]
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
    st.pyplot(fig)
