import numpy as np
import matplotlib.pyplot as plt


class Simulation:
    """Simulates particle interactions of colloidal substances"""

    def __init__(self, config) -> None:
        """Set up simulation"""
        self.size = config["size"]
        self.num_particles = config["num_particles"]
        self.particle_radius = config["particle_radius"]
        self.particle_distribution = config["particle_distribution"]
        self.timestep = config["timestep"]
        self.hydrodynamic_drag = config["hydrodynamic_drag"]
        self.brownian_amplitude = config["brownian_amplitude"]

        self.particles = None
        plt.ion()

        # force square aspect ratio so plot markers are accurate to
        # particle geometry
        self.fig = plt.figure(figsize=(5, 5))
        self.ax = self.fig.add_subplot(111)

    def add_particle(self, pos: tuple, mass: float, radius: float) -> None:
        """
        Adds a particle to the simulation's array of particles.

        Particle has no initial velocity as this is computed as brownian motion
        """
        vel_x, vel_y = 0, 0

        new_particle_data = np.array(
            [pos[0], pos[1], vel_x, vel_y, mass, radius])

        if self.particles is not None:
            self.particles = np.vstack((self.particles, new_particle_data))
        else:
            self.particles = np.ndarray(
                shape=(1, new_particle_data.shape[0]), buffer=new_particle_data)

    def generate_brownian_velocity(self, brownian_amplitude) -> np.ndarray:
        """Generates brownian motion velocity vector components"""
        # get number of particles
        n_particles = self.particles.shape[0]

        # compute random direction and magnitude
        d_headings = np.random.rand(n_particles,) * 2 * np.pi
        d_magnitude = np.random.rand(n_particles,) * brownian_amplitude

        # convert to cartesian delta velocity
        d_vel_x = d_magnitude * np.cos(d_headings)
        d_vel_y = d_magnitude * np.sin(d_headings)

        # zip component vectors into a matrix
        d_vel_brownian = np.stack((d_vel_x, d_vel_y), axis=-1)
        return d_vel_brownian

    def update_particles(self) -> None:
        """Updates particle positions"""
        # compute particle velocities
        d_vel_brownian = self.generate_brownian_velocity(
            self.brownian_amplitude)

        d_vel = d_vel_brownian
        d_vel = np.pad(d_vel, ((0, 0), (2, 2)), mode='constant',
                       constant_values=(0, 0))

        # update particle velocities
        self.particles = self.particles + d_vel
        self.particles[:, 2:4] *= self.hydrodynamic_drag

        # compute new particle positions
        vel = self.particles[:, 2:4]
        d_pos = vel * self.timestep
        d_pos = np.pad(d_pos, ((0, 0), (0, 4)), mode='constant',
                       constant_values=(0, 0))

        # update particle positions
        self.particles = self.particles + d_pos

        # Check for collisions between particles in their new positions
        for i in range(self.num_particles):
            for j in range(i+1, self.num_particles):
                pos_1 = self.particles[i, 0:2]
                pos_2 = self.particles[j, 0:2]

                dist = np.linalg.norm(pos_1 - pos_2)

                if dist <= 2 * self.particle_radius:
                    # Calculate the new velocities after the collision
                    vel_1 = self.particles[i, 2:4]
                    vel_2 = self.particles[j, 2:4]
                    mass_1 = self.particles[i, 4]
                    mass_2 = self.particles[j, 4]
                    vel_1_new = vel_1 - 2 * mass_2 / (mass_1 + mass_2) * np.dot(
                        vel_1 - vel_2, pos_1 - pos_2) / np.linalg.norm(pos_1 - pos_2) ** 2 * (pos_1 - pos_2)
                    vel_2_new = vel_2 - 2 * mass_1 / (mass_1 + mass_2) * np.dot(
                        vel_2 - vel_1, pos_2 - pos_1) / np.linalg.norm(pos_2 - pos_1) ** 2 * (pos_2 - pos_1)

                    # Update the velocities
                    self.particles[i, 2:4] = vel_1_new
                    self.particles[j, 2:4] = vel_2_new

                    # print("collision")

    def run(self) -> None:
        """Run and plot simulation"""
        # add particles
        if self.particle_distribution == "random":
            for _ in range(self.num_particles):
                self.add_particle(
                    (np.random.rand(2,) - .5) * self.size + self.size / 2,
                    1.,
                    self.particle_radius
                )
        # TODO: Distribute particles evenly on grid based on concentration
        # elif self.particle_distribution == "grid":
        #     it probably makes more sense to define particle positions with concentration and not number
        #     spacing = self.size / self.num_particles
        #     for x in range(self.num_particles):
        #         for y in range(self.num_particles // 2):
        #             self.add_particle((x * spacing, y * spacing),
        #                               1., self.particle_radius)

        # continually loop
        while True:
            self.update_particles()

            # clear graph and set limits
            self.ax.clear()
            self.ax.set_xlim([0, self.size])
            self.ax.set_ylim([0, self.size])

            # plot particles' positions and radii
            x = self.particles[:, 0]
            y = self.particles[:, 1]

            # calculate marker size on grid to match particle radius
            marker_radius = self.ax.transData.transform(
                [self.particle_radius, 0])[0] - self.ax.transData.transform([0, 0])[0]
            # TODO: this calculation is approximate and should be fixed
            marker_size = .5 * marker_radius**2

            self.ax.scatter(x, y, marker_size)

            # draw on canvas
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
