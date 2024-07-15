#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 23:06:58 2024

@author: sureshreddy
"""
import matplotlib.pyplot as plt
import meep as mp
import numpy as np
cell = mp.Vector3(16, 8, 0)

geometry = [
    mp.Block(
        mp.Vector3(mp.inf, 1, mp.inf),
        center=mp.Vector3(),
        material=mp.Medium(epsilon=12),
    )
]

sources = [
    mp.Source(
        mp.ContinuousSource(frequency=0.15), component=mp.Ez, center=mp.Vector3(-7, 0)
    )
]

pml_layers = [mp.PML(1.0)]

resolution = 10

sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=sources,
    resolution=resolution,
)

sim.run(until=200)




eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
plt.figure()
plt.imshow(eps_data.transpose(), interpolation="spline36", cmap="binary")
plt.axis("off")
plt.savefig("dielectric_function.png", bbox_inches='tight')
plt.show()

ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
# Compute the gradients of the Ez field
ez_grad_y, ez_grad_x = np.gradient(ez_data)
ez_grad_magnitude = np.sqrt(ez_grad_x**2 + ez_grad_y**2)

# Plot the Ez field
plt.figure()
plt.imshow(eps_data.transpose(), interpolation="spline36", cmap="binary", alpha=0.3)
plt.imshow(ez_data.transpose(), interpolation="spline36", cmap="RdBu", alpha=0.9)
plt.axis("off")
plt.title("Ez Field")
plt.colorbar()
plt.savefig("ez_field.png", bbox_inches='tight')
plt.show()

# Plot the gradient magnitude of the Ez field
plt.figure()
plt.imshow(eps_data.transpose(), interpolation="spline36", cmap="binary", alpha=0.3)
plt.imshow(ez_grad_magnitude.transpose(), interpolation="spline36", cmap="viridis", alpha=0.9)
plt.axis("off")
plt.title("Gradient Magnitude of Ez Field")
plt.colorbar()
plt.savefig("ez_gradient_magnitude.png", bbox_inches='tight')
plt.show()