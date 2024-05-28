Foundation Optimization Tool
This repository contains a tool for optimizing foundation geometry using Python and Streamlit. The tool calculates various parameters such as foundation weight, ballast and buoyancy, net vertical load, and pressures. It also includes functionality for optimizing the foundation dimensions to minimize concrete usage while maintaining structural integrity. The results are visualized using Plotly and Matplotlib.

Installation
Clone the repository:

sh
Copy code
git clone https://github.com/yourusername/foundation-optimization-tool.git
cd foundation-optimization-tool
Create and activate a virtual environment:

sh
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install dependencies:

sh
Copy code
pip install -r requirements.txt
Usage
To run the Streamlit application, execute the following command:

sh
Copy code
streamlit run app.py
Streamlit Interface
The interface consists of several sections where users can input parameters and visualize the results:

Load Cases: Input parameters for vertical force (F_z), resultant force (F_RES), moment (M_z), and resultant moment (M_RES).

Material Properties: Specify the material properties including maximum pressure (q_max), concrete density (rho_conc), wet ballast density (rho_ballast_wet), and dry ballast density (rho_ballast_dry).

Dimensions: Define the initial dimensions of the foundation components.

Calculation and Optimization:

Run Calculations: Calculates the initial foundation parameters and displays the results.
Optimize Foundation: Optimizes the foundation dimensions to reduce concrete usage and visualizes the optimized geometry.
Code Overview
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
Example Usage
Here's an example of how to use the tool:

Open the Streamlit interface by running:

sh
Copy code
streamlit run app.py
Enter the load cases, material properties, and initial dimensions in the sidebar.

Click the "Run Calculations" button to see the initial foundation parameters.

Click the "Optimize Foundation" button to optimize the foundation dimensions and visualize the results.

Contributing
Contributions are welcome! Please follow these steps to contribute:

Fork the repository.
Create a new branch with a descriptive name.
Make your changes.
Submit a pull request.
License
This project is licensed under the MIT License. See the LICENSE file for more details.

Contact
For any questions or feedback, please contact [your email address].

Feel free to customize this README to better suit your project's specific details and structure.
