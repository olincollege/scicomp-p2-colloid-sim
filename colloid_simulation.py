import numpy as np
import matplotlib.pyplot as plt


class Simulation:
    """Simulates particle interactions of colloidal substances"""

    def __init__(self, config) -> None:
        """Set up simulation"""
        self.width = config["width"]
        self.height = config["height"]
        self.num_particles = config["num_particles"]
        self.timestep = config["timestep"]
        self.brownian_amplitude = config["brownian_amplitude"]

        self.particles = None
        plt.ion()

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

    def add_particle(self, pos_x: float, pos_y: float, mass: float, radius: float) -> None:
        """
        Adds a particle to the simulation's array of particles.

        Particle has no initial velocity as this is computed as brownian motion
        """
        vel_x, vel_y = 0, 0

        new_particle_data = np.array(
            [pos_x, pos_y, vel_x, vel_y, mass, radius])

        if self.particles is not None:
            self.particles = np.vstack((self.particles, new_particle_data))
        else:
            self.particles = np.ndarray(
                shape=(1, new_particle_data.shape[0]), buffer=new_particle_data)

    def generate_brownian_velocity(self) -> np.ndarray:
        """Generates brownian motion velocity vector components"""
        # get number of particles
        n_particles = self.particles.shape[0]

        # compute random direction and magnitude
        d_headings = np.random.rand(n_particles,) * 2 * np.pi
        d_magnitude = np.random.rand(n_particles,) * self.brownian_amplitude

        # convert to cartesian delta velocity
        d_vel_x = d_magnitude * np.cos(d_headings)
        d_vel_y = d_magnitude * np.sin(d_headings)

        # zip component vectors into a matrix
        dv_brownian = np.stack((d_vel_x, d_vel_y), axis=-1)
        return dv_brownian

    def update_particles(self) -> None:
        """Updates particle positions"""
        # compute particle velocities
        d_vel_brownian = self.generate_brownian_velocity()
        d_vel = d_vel_brownian
        d_vel = np.pad(d_vel, ((0, 0), (2, 2)), mode='constant',
                       constant_values=(0, 0))

        # update particle velocities
        self.particles = self.particles + d_vel

        # compute new particle positions
        vel = self.particles[:, 2:4]
        d_pos = vel * self.timestep
        d_pos = np.pad(d_pos, ((0, 0), (0, 4)), mode='constant',
                       constant_values=(0, 0))

        # update particle positions
        self.particles = self.particles + d_pos

    def run(self) -> None:
        """Run and plot simulation"""
        # add particles
        for _ in range(self.num_particles):
            self.add_particle(128., 128., 1., 1.)

        # continually loop
        while True:
            self.update_particles()

            # clear graph and set limits
            self.ax.clear()
            self.ax.set_xlim([0, 256])
            self.ax.set_ylim([0, 256])

            # plot particles' positions and radii
            x = self.particles[:, 0]
            y = self.particles[:, 1]
            s = self.particles[:, 5]
            self.ax.scatter(x, y, s)

            # draw on canvas
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
