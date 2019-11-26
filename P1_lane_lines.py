#!/usr/bin/python3
# encoding: utf-8

# importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os

from moviepy.editor import VideoFileClip
from IPython.display import HTML


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def extrapolate(points, imshape):
    fit = np.polyfit(points[:, 0], points[:, 1], 1)
    top_y = int(imshape[0] * 0.6)
    top_x = int((top_y - fit[1]) / fit[0])
    bottom_y = imshape[0]
    bottom_x = int((bottom_y - fit[1]) / fit[0])

    return top_x, top_y, bottom_x, bottom_y


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    left_points = []
    right_points = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if (slope < -0.5) & (slope > -0.9):
                left_points.append([x1, y1])
                left_points.append([x2, y2])
            elif (slope > 0.4) & (slope < 0.8):
                right_points.append([x1, y1])
                right_points.append([x2, y2])

    if (len(left_points) == 0) | (len(right_points) == 0):
        return

    left_points = np.array(left_points)
    x1, y1, x2, y2 = extrapolate(left_points, img.shape)
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    right_points = np.array(right_points)
    x1, y1, x2, y2 = extrapolate(right_points, img.shape)
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.
def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


def image_draw_lines(img):
    gray = grayscale(img)

    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)

    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    imshape = img.shape
    vertices = np.array([[(0, imshape[0]), (470, 320),
                          (520, 320), (imshape[1], imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    rho = 2
    theta = np.pi / 180
    threshold = 50
    min_line_length = 100
    max_line_gap = 160
    line_img = hough_lines(masked_edges, rho, theta, threshold,
                           min_line_length, max_line_gap)
    lines_edges = weighted_img(line_img, img)

    return lines_edges


def image_test():
    for image in os.listdir('test_images/'):
        img = mpimg.imread('test_images/'+image)
        lines_edges = image_draw_lines(img)
        plt.imshow(lines_edges)
        plt.show()
        plt.imsave('test_images_output/'+image, lines_edges)


def process_image(img):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    lines_edges = image_draw_lines(img)
    return lines_edges


def video_test():
    for video in os.listdir('test_videos/'):
        clip1 = VideoFileClip("test_videos/" + video)
        white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
        white_output = 'test_videos_output/' + video
        white_clip.write_videofile(white_output, audio=False)


if __name__ == '__main__':
    image_test()
    video_test()
