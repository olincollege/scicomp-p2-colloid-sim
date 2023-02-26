from colloid_simulation import Simulation

# define simulation parameters
config = {
    "size": 50,
    "num_particles": 10,
    "particle_radius": 1.,
    "particle_distribution": "random",
    "timestep": .1,
    "hydrodynamic_drag": .9,
    "brownian_amplitude": .4
}

# create and run simulation
sim = Simulation(config)
sim.run()
