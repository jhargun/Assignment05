# Doing this to test git
# Doing more git testing
# What even is git stash

# Woah, git stash is really bizarre

import cv2
import os
import numpy as np

from Assignment05 import *

# I/O directories
input_dir = "inputs"
output_dir = "output"


def mp4_video_writer(filename, frame_size, fps=20):
    """Opens and returns a video for writing.

    Args:
        filename (string): Filename for saved video
        frame_size (tuple): Width, height tuple of output video
        fps (int): Frames per second
    Returns:
        VideoWriter: Instance of VideoWriter ready for writing
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    avi = filename.split('.')[0] + '.avi'
    print(avi)
    return cv2.VideoWriter(avi, fourcc, fps, frame_size)

def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    video = cv2.VideoCapture(filename)


    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    video.release()
    yield None;

# Helper code
def run_particle_filter(video_in, video_out, template_rect, num_particles, sigma_exp, sigma_dynamic, alpha = 0):

    # Initialize objects
    template = None
    pf = None
    frame_num = 0

    image_gen = video_frame_generator(video_in)
    frame = next(image_gen)

    #preparing output video
    h, w, d = frame.shape
    video_out = mp4_video_writer(video_out, (w, h), 20)

    #setup particle filter
    # print('Reached thing')
    pf = ParticleFilter(frame, template_rect, num_particles, sigma_exp, sigma_dynamic, alpha)
    # print('Passed thing')
    # Loop over video (till last frame or Ctrl+C is presssed)
    while frame is not None:
        # Process frame
        pf.process(frame)

        #save frame to video
        frame_out = frame.copy()
        pf.render(frame_out)
        video_out.write(frame_out)

        # Update frame number and move to next frame
        frame_num += 1
        if frame_num % 20 == 0:
            print('Working on frame %d' % frame_num)
        frame = next(image_gen)

    video_out.release()

def part_1():

    template_loc = {'y': 72, 'x': 140, 'w': 50, 'h': 50}

    num_particles, sigma_exp, sigma_dyn, alpha = get_input_values_for_filter(1)

    #num_particles = 100  # Define the number of particles
    #sigma_mse = 10  # Define the value of sigma for the measurement exponential equation
    #sigma_dyn = 10  # Define the value of sigma for the particles movement (dynamics)

    run_particle_filter(os.path.join(input_dir, "ball.avi"),
                        os.path.join(output_dir, "ball"),
                        template_loc,
                        num_particles = num_particles, sigma_exp = sigma_exp,
                        sigma_dynamic = sigma_dyn)


def part_2():

    template_loc = {'x': 360, 'y': 141, 'w': 127, 'h': 179}

    num_particles, sigma_exp, sigma_dyn, alpha = get_input_values_for_filter(2)

    #num_particles = 2000  # Define the number of particles
    #sigma_mse = 3  # Define the value of sigma for the measurement exponential equation
    #sigma_dyn = 10  # Define the value of sigma for the particles movement (dynamics)

    run_particle_filter(os.path.join(input_dir, "debate_noisy.avi"),
                        os.path.join(output_dir, "debate_noisy"),
                        template_loc,
                        num_particles=num_particles, sigma_exp=sigma_exp,
                        sigma_dynamic=sigma_dyn)


def part_3():
    template_rect = {'x': 538, 'y': 377, 'w': 73, 'h': 117}

    num_particles, sigma_exp, sigma_dyn, alpha = get_input_values_for_filter(3)

    #num_particles = 1000  # Define the number of particles
    #sigma_mse = 3 # Define the value of sigma for the measurement exponential equation
    #sigma_dyn = 30  # Define the value of sigma for the particles movement (dynamics)
    #alpha = 0.35  # Set a value for alpha

    run_particle_filter(os.path.join(input_dir, "debate.avi"),
                        os.path.join(output_dir, "debate"),
                        template_rect,
                        num_particles=num_particles, sigma_exp=sigma_exp,
                        sigma_dynamic=sigma_dyn, alpha=alpha)

def part_4():
    template_rect = {'x': 600, 'y': 55, 'w': 125, 'h': 162}

    num_particles, sigma_exp, sigma_dyn, alpha = get_input_values_for_filter(4)

    #num_particles = 1000  # Define the number of particles
    #sigma_mse = 3 # Define the value of sigma for the measurement exponential equation
    #sigma_dyn = 30  # Define the value of sigma for the particles movement (dynamics)
    #alpha = 0.00  # Set a value for alpha

    run_particle_filter(os.path.join(input_dir, "SNL.mp4"),
                        os.path.join(output_dir, "SNL"),
                        template_rect,
                        num_particles=num_particles, sigma_exp=sigma_exp,
                        sigma_dynamic=sigma_dyn, alpha=alpha)

if __name__ == '__main__':
    #comment out the parts you want to skip

    # part_1()  #tracking static object  ~100 frames
    # part_2()  #tracking static object  ~100 frames
    part_3()   #tracking changing object  ~160 frames
    # part_4()  #long tracking video  ~12,000 frames
