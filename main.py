from colloid_simulation import Simulation

# define simulation parameters
config = {
    "width": 256,
    "height": 256,
    "num_particles": 256,
    "timestep": 1.0,
    "brownian_amplitude": .1
}

# create and run simulation
sim = Simulation(config)
sim.run()
