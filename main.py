from colloid_simulation import Simulation

# define simulation parameters
config = {
    "size": 50,
    "num_particles": 50,
    "particle_radius": 1.,
    "particle_distribution": "random",
    "hydrodynamic_drag": .9,
    "brownian_amplitude": .4,
    "timestep": .1,
    "collision_check_mode": 'spatial_distance',  # or spatial_distance
    "show_grid": True
}

# create and run simulation
sim = Simulation(config)
sim.run()
