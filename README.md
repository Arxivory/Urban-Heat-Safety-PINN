# Urban-Aware Heat Safety: A Physics-Informed ML Approach

This project implements a **Physics-Informed Neural Network (PINN)** to estimate the Wet Bulb Globe Temperature (WBGT) in urban microclimates. By embedding the **Steady-State Energy Balance Equation** directly into the neural network's loss function, the model provides physically consistent heat risk assessments.

## The Problem
Standard heat indices often fail to account for the "Urban Heat Island" effect. In tropical urban environments—like the high-humidity street canyons of **Manila**—radiative heating from concrete and lack of wind can create dangerously high temperatures that standard weather apps miss.

## The Physics Model
The model predicts the Black Globe Temperature ($T_g$) by solving the energy balance:
$$Q_{shortwave} + Q_{longwave} + Q_{convection} = 0$$

Unlike "black-box" ML, this system is penalized if its predictions violate the Stefan-Boltzmann law or convective heat transfer coefficients.

## Features
- **Physics-Informed Loss:** Custom PyTorch loss function incorporating thermodynamic residuals.
- **Interactive Dashboard:** Real-time safety flag generation (White to Black flags) based on ISO 7243 standards.
- **Synthetic Data Pipeline:** A physics-based simulator for generating ground-truth thermal data.

## How to Use
1. Open the `urban_heat_safety_pinn.ipynb` in Google Colab.
2. Run the cells to train the model.
3. Use the interactive sliders at the bottom to simulate different weather scenarios.
