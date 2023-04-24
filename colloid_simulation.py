from time import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from matplotlib.cm import get_cmap


class Simulation:
    """Simulates particle interactions of colloidal substances"""

    def __init__(self, config) -> None:
        """Set up simulation"""
        self.size = config["size"]
        self.particle_concentration = config["particle_concentration"]
        self.particle_radius = config["particle_radius"]
        self.particle_mass = config["particle_mass"]
        self.particle_drag = config["particle_drag"]
        self.brownian_amplitude_initial = config["brownian_amplitude_initial"]
        self.brownian_amplitude_continuous = config["brownian_amplitude_continuous"]
        self.timestep = config["timestep"]
        self.show_grid = config["show_grid"]
        self.show_collisions = config["show_collisions"]
        self.show_velocities = config["show_velocities"]

        self.frame_count = 0
        self.particles = None
        self.particle_position_history = None
        self.history_max_steps = 500
        self.num_particles = 0
        self.bins = np.arange(
            0, self.size, self.particle_radius * 2)

        self.particle_to_show_path_index = None
        self.cmap_velocity = get_cmap('coolwarm')

        plt.ion()

        # force square aspect ratio so plot markers are accurate to
        # particle geometry
        self.fig = plt.figure(figsize=(5, 5), num="Particles Simulation")
        self.ax = self.fig.add_subplot(111,)

    def add_particle(self, pos: tuple, mass: float, radius: float) -> None:
        """
        Adds a particle to the simulation's array of particles

        Particle has no initial velocity as this is computed as brownian motion each time step
        """
        vel_x, vel_y = 0, 0

        new_particle_data = np.array(
            [pos[0], pos[1], vel_x, vel_y, mass, radius, 0])

        if self.particles is not None:
            self.particles = np.vstack((self.particles, new_particle_data))
        else:
            self.particles = np.ndarray(
                shape=(1, new_particle_data.shape[0]), buffer=new_particle_data)

    def build_hex_coords(self, x, y, width, height, spacing) -> np.ndarray:
        """
        Generate coordinates for a hex grid of particles

        Returns a 2D numpy array of coordinates of rows of particles
        hexagonally offset
        """
        # compute column and row spacing
        spacing_x = spacing
        spacing_y = spacing * np.sqrt(3)/2

        # generate list of points
        x_grid = np.arange(x+self.particle_radius,
                           width - self.particle_radius, spacing_x)
        y_grid = np.arange(y+self.particle_radius,
                           height - self.particle_radius, spacing_y)
        x_coords, y_coords = np.meshgrid(
            x_grid, y_grid, sparse=False, indexing='xy')

        # offset every other row
        x_coords[::2, :] += spacing/2

        coords = np.vstack((x_coords.flatten(), y_coords.flatten())).T
        return coords

    def generate_particle_grid(self) -> None:
        """Add particles in a hexagon grid pattern"""
        hex_coords = self.build_hex_coords(
            0, 0, self.size, self.size, self.particle_concentration)
        for coord in hex_coords:
            self.add_particle(coord, self.particle_mass, self.particle_radius)
            self.num_particles += 1

    def generate_brownian_velocity(self, brownian_amplitude) -> np.ndarray:
        """Generates brownian motion velocity vector components"""
        # get number of particles
        n_particles = self.particles.shape[0]

        # compute random direction and magnitude
        d_headings = np.random.rand(n_particles,) * 2 * np.pi
        d_magnitude = np.random.normal(size=n_particles,) * brownian_amplitude

        # convert to cartesian delta velocity
        d_vel_x = d_magnitude * np.cos(d_headings)
        d_vel_y = d_magnitude * np.sin(d_headings)

        # zip component vectors into a matrix
        d_vel_brownian = np.stack((d_vel_x, d_vel_y), axis=-1)
        return d_vel_brownian

    def update_particles(self) -> None:
        """Updates particle positions"""
        # compute new particle positions
        vel = self.particles[:, 2: 4]
        d_pos = vel * self.timestep

        # update particle positions
        self.particles[:, 0:2] += d_pos

    def resolve_collisions(self) -> None:
        """Resolve all particle collisions"""
        # reset flag specifying if particles might be colliding
        self.particles[:, 6] = 0

        # calculate pairwise distances with no repeats
        particle_positions = self.particles[:, 0:2]
        particle_distances = pdist(
            particle_positions)

        # convert to square form matrix for ease of indexing pairs
        particle_distances = squareform(particle_distances)

        # exclude particles distances from themselves (zero) by setting
        # their own distance to greater than the collision threshold
        particle_distances += np.identity(self.num_particles) * \
            (2 * self.particle_radius + 1)

        # check which particles are overlapping
        particle_bitmask = particle_distances <= (
            2 * self.particle_radius)

        # generate pairs based on truthy pairs of values
        particles_indices_pairs_to_check = np.transpose(
            np.nonzero(particle_bitmask))

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

            # offset particle positions so they're not overlapping in the next timestep
            # get distance required to unoverlap
            offset_dist = abs(np.linalg.norm(pos_1 - pos_2) -
                              2 * self.particle_radius)

            # compute direction in which to separate particles
            direction = pos_1 - pos_2

            # generate amount to offset particle positions by
            pos_offset_1 = direction * offset_dist / 2
            pos_offset_2 = direction * -offset_dist / 2

            # add particle positions
            self.particles[index_pair[0], 0:2] += pos_offset_1
            self.particles[index_pair[1], 0:2] += pos_offset_2

        # check for collisions with the walls
        collisions_x_min = np.where(
            particle_positions[:, 0] < self.particle_radius)
        collisions_x_max = np.where(
            particle_positions[:, 0] > self.size - self.particle_radius)
        collisions_y_min = np.where(
            particle_positions[:, 1] < self.particle_radius)
        collisions_y_max = np.where(
            particle_positions[:, 1] > self.size - self.particle_radius)

        # reverse the velocities of particles that hit the walls
        self.particles[collisions_x_min, 2] *= -1
        self.particles[collisions_x_max, 2] *= -1
        self.particles[collisions_y_min, 3] *= -1
        self.particles[collisions_y_max, 3] *= -1

        # offset particle positions so they're not intersecting walls
        self.particles[collisions_x_min, 0] = self.particle_radius
        self.particles[collisions_x_max, 0] = self.size - self.particle_radius

        self.particles[collisions_y_min, 1] = self.particle_radius
        self.particles[collisions_y_max, 1] = self.size - self.particle_radius

    def apply_brownian_velocity(self, brownian_amplitude) -> None:
        """Applies brownian velocity component to particles"""

        self.particles[:, 2:4] += self.generate_brownian_velocity(
            brownian_amplitude)

    def apply_particle_drag(self) -> None:
        """Reduces particle velocity by a set multiplier"""
        self.particles[:,
                       2: 4] *= np.clip(1 - self.particle_drag, 0., 1.)

    def clear_graph(self) -> None:
        """Clear graph and set limits"""
        self.ax.clear()
        self.ax.set_xlim([0, self.size])
        self.ax.set_ylim([0, self.size])

    def compute_marker_size(self) -> int:
        """Calculate marker size on grid to match particle radius"""
        marker_radius = self.ax.transData.transform(
            [self.particle_radius, 0])[0] - self.ax.transData.transform([0, 0])[0]
        marker_size = .5 * marker_radius**2
        return marker_size

    def run(self) -> None:
        """Run and plot simulation"""

        self.generate_particle_grid()

        # add first positions of particles to history store
        self.particle_position_history = [self.particles[:, 0:2]]

        # apply initial random particle direction
        self.apply_brownian_velocity(self.brownian_amplitude_initial)

        # start fps counter for initial run
        frame_time_start = time()

        # continuously loop to run simulation
        while True:
            ### POSITION CALCULATIONS ###

            # apply kinematics continuously
            self.apply_particle_drag()
            self.apply_brownian_velocity(self.brownian_amplitude_continuous)

            # update particles based on kinematics alone
            self.update_particles()

            # resolve collisions
            self.resolve_collisions()

            ### PLOTTING ###
            self.clear_graph()

            marker_size = self.compute_marker_size()

            # get particles' positions
            x = self.particles[:, 0]
            y = self.particles[:, 1]

            # color code particles by velocity
            if self.show_velocities:
                vels = self.particles[:, 2:4]
                mags = np.sqrt(vels[:, 0]**2 + vels[:, 1]**2)
                particle_color = self.cmap_velocity(mags)
                particle_color[:, 3] = .5  # set alpha
            else:
                # else set to blue
                blue = [0., 0., 1., .5]
                particle_color = np.tile(blue, (self.num_particles, 1))

            # store history of particles
            if self.frame_count % 2 == 0:
                last_particle_positions = self.particles[:, 0:2]
                self.particle_position_history = np.concatenate(
                    (self.particle_position_history, [last_particle_positions]))

            # set particle color if it's being checked for collisions
            if self.show_collisions:
                colliding_particle_indices = np.where(self.particles[:, 6])
                red = [1., 0., 0., .5]
                particle_color[colliding_particle_indices] = red

            # only keep a set amount as to not fill memory
            if self.particle_position_history.shape[0] > self.history_max_steps - 1:
                self.particle_position_history = self.particle_position_history[1:]

            # select particle index when it's clicked
            def onpick(event):
                ind = event.ind
                self.particle_to_show_path_index = ind[0]

            # plot past positions of clicked particle as line and color in that particle
            if self.particle_to_show_path_index is not None:
                # set opacity of clicked particle marker to 100%
                particle_color[self.particle_to_show_path_index] = [
                    0., 0., 0., .5]
                particle_to_show_path = self.particle_position_history[:,
                                                                       self.particle_to_show_path_index, :]

                self.ax.plot(
                    particle_to_show_path[:, 0],
                    particle_to_show_path[:, 1],
                    color='black',
                    alpha=.75
                )

            # plot particles
            self.ax.scatter(x, y,
                            s=marker_size, color=particle_color,
                            edgecolors='none', picker=True)

            # compute and show FPS
            frame_time_end = time()
            fps = 1 / (frame_time_end - frame_time_start)
            text_fps_string = f'FPS: {fps:.1f}'
            text_fps = self.fig.text(.12, .025, text_fps_string, fontsize=10)

            # show number of particles
            text_num_particles_string = f'Particles: {self.num_particles}'
            text_num_particles = self.fig.text(.4, .025,
                                               text_num_particles_string, fontsize=10)

            # show number of current collisions
            num_collisions = np.count_nonzero(self.particles[:, 6])
            text_num_collisions_string = f'Collisions: {num_collisions}'
            text_num_collisions = self.fig.text(.68, .025,
                                                text_num_collisions_string, fontsize=10)

            # draw on canvas
            self.fig.canvas.mpl_connect('pick_event', onpick)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            # reset fps timer
            frame_time_start = time()

            # remove old text
            text_fps.remove()
            text_num_particles.remove()
            text_num_collisions.remove()

            self.frame_count += 1
