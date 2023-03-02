from colloid_simulation import Simulation

# define simulation parameters
config = {
    "size": 50,
    "particle_concentration": 2.5,
    "particle_radius": 1.,
    "particle_mass": 1.,
    "hydrodynamic_drag": .9,
    "brownian_amplitude": .4,
    "timestep": .1,
    "collision_check_mode": 'spatial_distance',  # or "bounding_boxes"
    "show_grid": True,
    "show_collisions": True
}

# create and run simulation
sim = Simulation(config)
sim.run()
