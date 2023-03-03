from colloid_simulation import Simulation

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

# create and run simulation
sim = Simulation(config)
sim.run()
