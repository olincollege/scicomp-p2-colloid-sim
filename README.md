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
    "num_particles": 50,
    "particle_radius": 1.,
    "particle_distribution": "random",
    "hydrodynamic_drag": .9, # 1.0 is no drag
    "brownian_amplitude": .4,
    "timestep": .1,
    "collision_check_mode": 'bounding_boxes',  # or spatial_distance
    "show_grid": True
}
```

Play around with these to control how the simulation behaves!

## Running
To start the simulation as configured, run:

```python main.py```

This opens a matplotlib window. Closing this will terminate the simulation and return you to your prompt.