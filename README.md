# Colloid Simulation
A simple simulation of colloidal substances, modeled through the collisions of hard spheres colliding in a solvent.

## Setup
You'll need at least Python 3.9 and a few packages to run this simulation. To install them, run:

```pip install -r requirements.txt```

## Customization
In `main.py`, you'll find a few different parameters defined in the `config` object:

```python
# define simulation parameters
config = {
    "size": 50,
    "particle_concentration": 2.5,
    "particle_radius": 1.,
    "particle_mass": 1.,
    "hydrodynamic_drag": 0.,
    "brownian_amplitude_initial": 1,
    "brownian_amplitude_continuous": 0,
    "timestep": .1,
    "show_grid": False,  # requires significant render time
    "show_collisions": True,
    "show_velocities": True
}
```

Play around with these to control how the simulation behaves!

## Running
To start the simulation as configured, run:

```python main.py```

This opens a matplotlib window. Closing this will terminate the simulation and return you to your prompt.