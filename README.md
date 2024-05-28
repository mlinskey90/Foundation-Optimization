# Foundation Optimization App

This repository contains a Streamlit app for optimizing foundation dimensions and weights. The app allows users to input various parameters related to the foundation and then run calculations or optimize the foundation dimensions to minimize the total weight. The results are displayed along with a comparison plot of the original and optimized foundation shapes.

## Features

- Calculate foundation dimensions and weights based on user inputs.
- Optimize foundation dimensions to minimize total weight.
- Visualize the original and optimized foundation shapes with a comparison plot.

## Requirements

- Python 3.6 or higher
- Streamlit
- NumPy
- SciPy
- Plotly
- Matplotlib

  ## Usage
  To run the Streamlit application, execute the following command:
   ```bash
   streamlit run app.py
## Streamlit Interface
Streamlit Interface
The interface consists of several sections where users can input parameters and visualize the results:

1. Load Cases: Input parameters for vertical force (F_z), resultant force (F_RES), moment (M_z), and resultant moment (M_RES).

2. Material Properties: Specify the material properties including maximum pressure (q_max), concrete density (rho_conc), wet ballast density (rho_ballast_wet), and dry ballast density (rho_ballast_dry).

3. Dimensions: Define the initial dimensions of the foundation components.

4. Calculation and Optimization:

   Run Calculations: Calculates the initial foundation parameters and displays the results.
   Optimize Foundation: Optimizes the foundation dimensions to reduce concrete usage and visualizes the optimized        geometry.

## Code Overview
The main components of the code include:

Function Definitions:

calculate_foundation_weight: Calculates the weight of different parts of the foundation.
calculate_ballast_and_buoyancy: Computes ballast and buoyancy forces.
net_vertical_load: Determines the net vertical load on the foundation.
calculate_pressures: Calculates the minimum and maximum pressures.
plot_foundation_comparison: Plots a comparison of the original and optimized foundation geometries.
run_calculations: Runs the calculations and returns the results.
optimize_foundation: Optimizes the foundation dimensions and returns the results.
plot_3d_foundation: Plots the 3D representation of the foundation geometry.
Streamlit Interface:

Input fields for load cases, material properties, and dimensions.
Buttons for running calculations and optimizing the foundation.
Displays results and visualizations.
## Installation

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
2. **Create and Activate a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. **Install Dependencies**   
```bash
   pip install -r requirements.txt
