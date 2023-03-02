from collections import defaultdict
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


class Simulation:
    """Simulates particle interactions of colloidal substances"""

    def __init__(self, config) -> None:
        """Set up simulation"""
        self.size = config["size"]
        self.num_particles = config["num_particles"]
        self.particle_radius = config["particle_radius"]
        self.particle_distribution = config["particle_distribution"]
        self.hydrodynamic_drag = config["hydrodynamic_drag"]
        self.brownian_amplitude = config["brownian_amplitude"]
        self.timestep = config["timestep"]
        self.collision_check_mode = config["collision_check_mode"]
        self.show_grid = config["show_grid"]

        self.particles = None
        self.bins = np.arange(
            0, self.size, self.particle_radius * 2)

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
            [pos[0], pos[1], vel_x, vel_y, mass, radius, 0])

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
        # compute new particle positions
        vel = self.particles[:, 2:4]
        d_pos = vel * self.timestep
        d_pos = np.pad(d_pos, ((0, 0), (0, 5)), mode='constant',
                       constant_values=(0, 0))

        # update particle positions
        self.particles = self.particles + d_pos

    def apply_brownian_velocity(self) -> None:
        """Applies brownain velocity component to particles"""

        self.particles[:, 2:4] += self.generate_brownian_velocity(
            self.brownian_amplitude)

    def apply_hydrodynamic_drag(self) -> None:
        """Reduces particle velocity by a set multiplier"""
        self.particles[:, 2:4] *= self.hydrodynamic_drag

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
        # TODO: Distribute particles evenly on grid based on concentration without overlapping
        # elif self.particle_distribution == "grid":
        #     it probably makes more sense to define particle positions with concentration and not number
        #     spacing = self.size / self.num_particles
        #     for x in range(self.num_particles):
        #         for y in range(self.num_particles // 2):
        #             self.add_particle((x * spacing, y * spacing),
        #                               1., self.particle_radius)

        # continually loop
        while True:
            ### POSITION CALCULATION ###
            self.apply_hydrodynamic_drag()
            self.apply_brownian_velocity()

            # update particles based on kinematics alone
            self.update_particles()

            # reset flag specifying if particles might be colliding
            self.particles[:, 6] = 0

            # TODO: this currently only checks if particle centers are in the same grid
            # square and misses many intersections when they are not.
            if self.collision_check_mode == "bounding_boxes":
                # quantize (digitize) particles into bins
                digitized_particle_coords = np.digitize(
                    self.particles[:, 0:2] - self.particle_radius, self.bins, right=True)

                # NOTE: unused but could be used for density calculations
                # hist = np.histogram2d(
                #     digitized_particle_coords[:, 0], digitized_particle_coords[:, 1])[0] // 1 - 1
                # hist_collisions = hist != 0

                # compute actual positions of bins and plot
                binned_particle_pos = digitized_particle_coords * 2 * self.particle_radius

                # create dict with keys as binned coordinates and values as a lists of particles in those bins
                binned_particles = defaultdict(list)
                for i, digitized_particle_coord in enumerate(digitized_particle_coords):
                    binned_particles[tuple(
                        digitized_particle_coord)].append(i)

                # filter for only bins with multiple particles
                binned_particles_to_check = {
                    k: v for k, v in binned_particles.items() if len(v) != 1}

                # compute collision for sets of particles that are close enough to have them
                for particles_indices in binned_particles_to_check:
                    # generate combinations of particle indices to check
                    index_pairs = list(combinations(particles_indices, 2))
                    for index_pair in index_pairs:
                        particle_1 = self.particles[index_pair[0], :]
                        particle_2 = self.particles[index_pair[1], :]

                        pos_1 = particle_1[0:2]
                        pos_2 = particle_2[0:2]

                        # compute distance between particles
                        dist = np.linalg.norm(pos_1 - pos_2)

                        if dist <= 2 * self.particle_radius:
                            # Calculate the new velocities after the collision
                            vel_1 = particle_1[2:4]
                            vel_2 = particle_2[2:4]
                            mass_1 = particle_1[4]
                            mass_2 = particle_2[4]

                            vel_1_new = vel_1 - 2 * mass_2 / (mass_1 + mass_2) * np.dot(
                                vel_1 - vel_2, pos_1 - pos_2) / np.linalg.norm(pos_1 - pos_2) ** 2 * (pos_1 - pos_2)
                            vel_2_new = vel_2 - 2 * mass_1 / (mass_1 + mass_2) * np.dot(
                                vel_2 - vel_1, pos_2 - pos_1) / np.linalg.norm(pos_2 - pos_1) ** 2 * (pos_2 - pos_1)

                            # update particle velocities
                            self.particles[index_pair[0], 2:4] = vel_1_new
                            self.particles[index_pair[1], 2:4] = vel_2_new

                # generate a flattened list of particles that could be colliding for visualization purposes
                all_colliding_particle_indices = list(
                    binned_particles_to_check.values())
                all_colliding_particle_indices = [
                    i for overlapping in all_colliding_particle_indices for i in overlapping]

                # set flag if particles might be colliding
                self.particles[all_colliding_particle_indices, 6] = 1

            elif self.collision_check_mode == "spatial_distance":
                # calculate pairwise distances with no repeats
                particle_positions = self.particles[:, 0:2]
                particle_distances = pdist(
                    particle_positions)

                # convert to square form matrix for ease of indexing pairs
                particle_distances = squareform(particle_distances)

                # exclude particles distances from themselves (zero) by setting their own distance to greater than the collision threshold
                particle_distances += np.identity(self.num_particles) * \
                    (2 * self.particle_radius + 1)

                # check which particles are overlapping
                particle_bitmask = particle_distances <= (
                    2 * self.particle_radius)

                # TODO: set the diagonal half of the matrix false here

                # generate pairs based on truthy pairs of values
                particles_indices_pairs_to_check = np.transpose(
                    np.nonzero(particle_bitmask))

                # TODO: instead of this, only use one diagonal half of matrix to avoid duplicate checks
                particles_indices_pairs_to_check = [
                    tuple(sorted(pair)) for pair in particles_indices_pairs_to_check]
                particles_indices_pairs_to_check = list(set(
                    particles_indices_pairs_to_check))

                # generate list of all potentially colliding particles
                all_colliding_particle_indices = [
                    i for sublist in particles_indices_pairs_to_check for i in sublist]
                all_colliding_particle_indices = list(
                    set(all_colliding_particle_indices))

                # set flag if particles might be colliding for visualization
                self.particles[all_colliding_particle_indices, 6] = 1

                # compute collision for sets of particles that are close enough to have them
                for index_pair in particles_indices_pairs_to_check:
                    particle_1 = self.particles[index_pair[0], :]
                    particle_2 = self.particles[index_pair[1], :]

                    # get necessary attributes for collision computation
                    pos_1 = particle_1[0:2]
                    pos_2 = particle_2[0:2]
                    vel_1 = particle_1[2:4]
                    vel_2 = particle_2[2:4]
                    mass_1 = particle_1[4]
                    mass_2 = particle_2[4]

                    # calculate the new velocities after the collision
                    vel_1_new = vel_1 - 2 * mass_2 / (mass_1 + mass_2) * np.dot(
                        vel_1 - vel_2, pos_1 - pos_2) / np.linalg.norm(pos_1 - pos_2) ** 2 * (pos_1 - pos_2)
                    vel_2_new = vel_2 - 2 * mass_1 / (mass_1 + mass_2) * np.dot(
                        vel_2 - vel_1, pos_2 - pos_1) / np.linalg.norm(pos_2 - pos_1) ** 2 * (pos_2 - pos_1)

                    # update particle velocities
                    self.particles[index_pair[0], 2:4] = vel_1_new
                    self.particles[index_pair[1], 2:4] = vel_2_new

            # plot histogram of particle density per bin
            # NOTE: unused and unscaled
            # self.ax.matshow(hist * self.particle_radius)

            ### PLOTTING ###

            # clear graph and set limits
            self.ax.clear()
            self.ax.set_xlim([0, self.size])
            self.ax.set_ylim([0, self.size])

            # plot gridlines corresponding to particle bins
            if self.show_grid:
                self.ax.set_yticks(self.bins+self.particle_radius, minor=True)
                self.ax.set_xticks(self.bins+self.particle_radius, minor=True)
                self.ax.grid(which='minor', linestyle='--', linewidth=0.5)

            # get particles' positions and radii
            x = self.particles[:, 0]
            y = self.particles[:, 1]

            # calculate marker size on grid to match particle radius
            marker_radius = self.ax.transData.transform(
                [self.particle_radius, 0])[0] - self.ax.transData.transform([0, 0])[0]
            # TODO: this calculation is approximate and should be fixed
            marker_size = .42 * marker_radius**2

            # set particle color if it's being checked for collisions
            particle_color = [
                'orange' if state == 1 else 'blue' for state in self.particles[:, 6]]

            # plot particles
            self.ax.scatter(x, y, marker_size, color=particle_color, alpha=.25)

            # plot binned particles
            # self.ax.scatter(
            #     binned_particle_pos[:, 0], binned_particle_pos[:, 1], marker_size, color='g', marker='x')

            # draw on canvas
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
