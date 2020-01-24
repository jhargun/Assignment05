import numpy as np
import cv2

from math import ceil, exp

def get_input_values_for_filter(part_number):
    """A function to get the values for various constant values for the
    different parts of this assignment

    Args:
        part_number (int):  which part of the project we're getting values for, in [1, 4]
    Returns:
        tuple of numbers: (num_particles, sigma_exponential, sigma_dynamic, alpha)
    """
    if part_number == 1:
        num_particles = 100
        sigma_exponential = 10
        sigma_dynamic = 15
        alpha = 0
    elif part_number == 2:
        num_particles = 2000
        sigma_exponential = 3
        sigma_dynamic = 10
        alpha = 0
    elif part_number == 3:
        num_particles = 1000
        sigma_exponential = 3
        sigma_dynamic = 30
        alpha = .1  # Original (.35) led to it detecting too far to the left because template was updating too fast
    elif part_number == 4:
        num_particles = 1000
        sigma_exponential = 3
        sigma_dynamic = 30
        alpha = 0
    return (num_particles, sigma_exponential, sigma_dynamic, alpha)



class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in tests.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template_rect, num_particles, sigma_exp, sigma_dyn, alpha = 0):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one).
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template_rect (dict):  Template coordinates with x, y,
                                            width, and height values.
            num_particles (int): number of particles.
            sigma_exp (float): sigma value used in the similarity measure.
            sigma_dyn (float): sigma value that can be used when adding gaussian noise to particle positions.
            alpha (float):  value used to determine how much we adjust our template
        """
        self.sigma_exp = sigma_exp
        self.sigma_dyn = sigma_dyn
        self.alpha = alpha
        t_X = template_rect['x']
        t_Y = template_rect['y']
        height = template_rect['h']
        width = template_rect['w']
        self.template = frame[t_Y:t_Y+height, t_X:t_X+width]
        # cv2.imshow("template", self.template)  # Shows template, useful if debugging
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        x = np.random.uniform(t_X, t_X+width, num_particles)  # Uniform distribution over template
        y = np.random.uniform(t_Y, t_Y+height, num_particles)
        self.particles = np.vstack((x, y)).T.astype(int)
        self.weights = np.ones(num_particles) / num_particles  # Makes uniform weights

    def get_likelyhood(self, template, frame_cutout):
        """Returns the likelyhood (probability) measure of observing the template at the given
            frame_cutout location.  We will do this by calculating the mean squared difference between
            the template we're tracking and the frame_cutout, then we use an exponential function
            to convert that value to a probability
            You should try to calculate exp(-1*meanSqDiff / 2 * sigma^2) and return that value
        Returns:
            float: likelyhood value
        """
        # This assumes frame_cutout is the actual cutout, not just the location.
        msdif = np.mean((self.template.astype(np.float32) - frame_cutout.astype(np.float32)) ** 2)
        return exp(-1*msdif / (2 * self.sigma_exp ** 2))  # I think sigma is meant to be in denominator

    def resample_particles(self):
        """Returns a new set of particles

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.

        Returns:
            numpy.array: particles data structure.
        """
        new_parts_x = np.random.choice(self.particles[:, 0], self.particles.shape[0], p=self.weights)  # Makes new particles using old weights
        new_parts_y = np.random.choice(self.particles[:, 1], self.particles.shape[0], p=self.weights)
        new_parts = np.vstack((new_parts_x, new_parts_y)).T
        self.weights = np.ones(len(self.particles)) / len(self.particles)  # Change weights to uniform distribution
        return new_parts


    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Your general algorithm should look something like this:
        resample particles
        for each particle:
            disperse it using sigma_dyn
            get the cutout corresponding to the particle
            calculate the likelyhood of the cutout based on the template
            change the weight of the particle
        normalize the weights
        update the template based on alpha (not needed for part 1, 2)

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        self.particles = self.resample_particles() + np.random.normal(0, self.sigma_dyn, self.particles.shape)  # Randomizes positions
        self.particles = self.particles.astype(int)

        width = self.template.shape[1]  # Width and height of template
        height = self.template.shape[0]
        f_w = frame.shape[1]  # Width and height of frame
        f_h = frame.shape[0]

        # Next part: in case any particles went out, put them into random position
        # All particles need to be some space away from edge since they're detecting center of template
        w_2 = ceil(width/2)  # Minimum distance of particles from left and right sides
        h_2 = ceil(height/2)  # Distance from top and bottom
        rand_parts_h = np.random.randint(h_2, f_h-h_2, self.particles.shape[0])
        rand_parts_w = np.random.randint(w_2, f_w-w_2, self.particles.shape[0])
        rand_parts = np.vstack((rand_parts_w, rand_parts_h)).T

        # These next 4 lines replace any out of bounds particles with random particles
        self.particles = np.where(np.array([self.particles[:, 0] < w_2]*2).T, rand_parts, self.particles)
        self.particles = np.where(np.array([self.particles[:, 1] < h_2]*2).T, rand_parts, self.particles)
        self.particles = np.where(np.array([self.particles[:, 0] > f_w-w_2]*2).T, rand_parts, self.particles)
        self.particles = np.where(np.array([self.particles[:, 1] > f_h-h_2]*2).T, rand_parts, self.particles)

        for i in range(len(self.particles)):  # Change weights
            x = self.particles[i][0] - w_2
            y = self.particles[i][1] - h_2
            likelyhood = self.get_likelyhood(self.template, frame[y:y+height, x:x+width])
            self.weights[i] = self.weights[i] * likelyhood

        # Normalize weights
        self.weights = self.weights / np.sum(self.weights)

        # Updating template using alpha
        x_av = np.average(self.particles[:, 0], weights=self.weights)
        y_av = np.average(self.particles[:, 1], weights=self.weights)
        x = int(x_av - w_2)  # X and y values of the top left of the rectangle
        y = int(y_av - h_2)
        predicted_template = frame[y:y+height, x:x+width]
        self.template = self.template * (1-self.alpha) + predicted_template * self.alpha


    def render(self, frame_in):
        """Visualizes current particle filter state.

        Don't do any model updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius (Make it a color that will standout!).
        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        Returns:
            Nothing, but you should do all of your drawing on the frame_in
        """
        for p in self.particles:  # Shows particles
            p = tuple(p.astype(int))
            cv2.circle(frame_in, p, 2, (255,0,0), -1)

        # Getting weighted mean and stdev of x and y values
        x_av = np.average(self.particles[:, 0], weights=self.weights)
        y_av = np.average(self.particles[:, 1], weights=self.weights)

        width = self.template.shape[1]  # Width and height of template
        height = self.template.shape[0]

        x = int(x_av - width//2)  # X and y values of the top left of the rectangle
        y = int(y_av - height//2)

        cv2.rectangle(frame_in, (x, y), (x+width, y+height), 255, 2)  # Makes box for predicted location


        # Making the circle
        distances = np.sqrt(np.add(np.square(self.particles[:, 0] - x_av), np.square(self.particles[:, 1] - y_av)))  # Pythagorean theorem
        radius = np.sum(np.multiply(distances, self.weights))  # Weighted sum
        cv2.circle(frame_in, (int(x_av), int(y_av)), int(radius), (0, 0, 255), 2)
