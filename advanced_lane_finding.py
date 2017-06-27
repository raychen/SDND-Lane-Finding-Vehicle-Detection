# -*- coding: utf8 -*-
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys
from glob import glob
from tqdm import tqdm_notebook, tqdm
from moviepy.editor import VideoFileClip


def get_objp(nx, ny):
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    return objp


def findChessboardCornes(path_pattern, nx=9, ny=6):
    objp = get_objp(nx, ny)

    objpoints = []
    imgpoints = []

    for filename in tqdm(glob(path_pattern)):
        img = mpimg.imread(filename)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray_img, (nx, ny), None)

        if ret == True:
            # Draw and display the corners
            #         cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            objpoints.append(objp)
            imgpoints.append(corners.reshape(nx * ny, 2))

    print("%d images were proccssed" % len(imgpoints))

    return objpoints, imgpoints


def get_transformation_matrix(src, dst):
    trans_m = cv2.getPerspectiveTransform(src, dst)
    trans_m_inv = cv2.getPerspectiveTransform(dst, src)
    return trans_m, trans_m_inv


def threshold_binary(test_img, cvtColor=None, ch_idx=0, low=0, hi=255):
    img_size = (test_img.shape[1], test_img.shape[0])


    # filter color channel
    if cvtColor is not None:
        test_img = cv2.cvtColor(test_img, cvtColor)

    channel_img = test_img[:, :, ch_idx]
    binary = np.zeros_like(test_img[:, :, ch_idx])
    binary[(channel_img > low) & (channel_img <= hi)] = 1
    return binary


def visualize_channel_threshold(test_img, trans_m, threshold_base=0, color_space='HLS'):
    """
    use a 5x5 grid to visualize different low-high threshold range
    :param test_img:
    :param trans_m: transformation matrix
    :param threshold_base: minimum threshold to start search
    :param color_space:
    :return:
    """
    print("Color Space:", color_space)
    num_channels = 3
    plt.figure(figsize=(8, 15))
    if color_space is 'HLS':
        test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2HLS)
    elif color_space is 'HSV':
        test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2HSV)
    elif color_space is 'YUV':
        test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2YUV)

    warped_img = cv2.warpPerspective(test_img, trans_m, img_size, flags=cv2.INTER_LINEAR)
    for ch_idx in range(num_channels):
        base_plot = ch_idx * 25
        num_rows = 5 * num_channels
        num_cols = 5
        step = (255 - threshold_base) / 5
        for lo in range(5):
            base_plot += lo
            for hi in range(lo + 1, 6):
                base_plot += 1
                low = int(lo * step) + threshold_base
                high = int(hi * step) + threshold_base
                ax = plt.subplot(num_rows, num_cols, base_plot)
                plt.setp(ax.get_xticklines(), visible=False)
                plt.setp(ax.get_yticklines(), visible=False)
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
                b = threshold_binary(warped_img, ch_idx=ch_idx, low=low, hi=high)
                plt.title("%d - %d|%d" % (low, high, b.sum()), fontsize=6)
                plt.imshow(b)


class Line(object):
    def __init__(self, use_best_fit=True, keep=10):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = []
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')

        # raw x, y values before fitting
        self.rawx = None
        self.rawy = None

        # fitted x, y, ready for drawing
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

        # how many previous fit params to keep
        self.keep = keep

        #
        self.use_best_fit = use_best_fit

    def calculate_best_fit(self, num=10):
        if len(self.current_fit) == 0:
            return
        self.best_fit = np.mean(np.array(self.current_fit[-num:]), axis=0)

    def calculate_fit_diff(self, new_fit):
        # diff between new fit and best fit
        if self.best_fit is None:
            return None
        return new_fit - self.best_fit

    def fit_line(self, y_range):
        if self.rawx is None or len(self.rawx) == 0:
            return False

        new_fit = np.polyfit(self.rawy, self.rawx, 2)
        diff = self.calculate_fit_diff(new_fit)
        # only use this new fit if diff smaller than 0.1
        if self.use_best_fit is False or diff is None or diff[0] < 0.1:
            self.current_fit.append(new_fit)
            self.calculate_best_fit(num=self.keep)
            self.recent_xfitted, _ = self.get_fitted_pts(y_range)
            return True
        else:
            return False

    def add_raw_pts(self, x, y, new_y_range):
        self.rawx = x
        self.rawy = y
        return self.fit_line(new_y_range)

    def get_fitted_pts(self, y_range):
        y = np.arange(y_range[0], y_range[1])
        # use best fit
        fit = self.best_fit if self.use_best_fit else self.current_fit[-1]
        fit_x = fit[0] * y ** 2 + fit[1] * y + fit[2]

        return fit_x, y

    def calculate_radius_and_midpoint(self, other, y_eval, ym_per_pix=30 / 720, xm_per_pix=3.7 / 700):
        y = np.arange(0, y_eval)

        self_fit = np.polyfit(y * ym_per_pix, self.recent_xfitted * xm_per_pix, 2)
        other_fit = np.polyfit(y * ym_per_pix, other.recent_xfitted * xm_per_pix, 2)

        avg_fit_cr = np.mean([self_fit, other_fit], axis=0)
        avg_curverad = ((1 + (2 * avg_fit_cr[0] * y_eval * ym_per_pix + avg_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * avg_fit_cr[0])

        midpoint = (self.recent_xfitted[-1] + other.recent_xfitted[-1]) // 2
        return avg_curverad, midpoint


def region_select(img, lo_left, hi_left, lo_right, hi_right, fill=(255, 255, 255)):
    xx, yy = np.meshgrid(np.arange(0, img.shape[1]), np.arange(0, img.shape[0]))
    left_edge = np.polyfit([lo_left[0], hi_left[0]], [lo_left[1], hi_left[1]], 1)
    right_edge = np.polyfit([lo_right[0], hi_right[0]], [lo_right[1], hi_right[1]], 1)

    threshold = (yy > left_edge[0] * xx + left_edge[1]) | \
                (yy > right_edge[0] * xx + right_edge[1])
    img[threshold] = fill
    return img


def stack_channel_binary(warped_img, thresholds, extra_channel=None):
    # thresholds format: (cvtColor, ch_idx, low, hi, max_sum)

    stacked = np.zeros_like(warped_img[:, :, 0])
    for t in thresholds:
        b = threshold_binary(warped_img, cvtColor=t[0], ch_idx=t[1], low=t[2], hi=t[3])
        # too many points are 1 indicating this binary is not useful
        if b.sum() <= t[4]:
            stacked[b == 1] = 1

    if extra_channel is not None:
        for c in extra_channel:
            b = c(warped_img)
            stacked[b == 1] = 1

    # remove bottom 4% row of image to get rid of the car hood
    stacked[int(stacked.shape[0] * .96):, :] = 0
    return stacked

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def lane_histogram(binary_warped, dpi=216):
    # prepare figure and canvas to draw on
    fig = plt.gcf()
    fig.dpi = dpi
    canvas = FigureCanvas(fig)
    plt.xlim = (0, 1200)
    plt.ylim = (0, 720)

    # plot warped binary
    ax1 = plt.subplot(2, 1, 1)
    ax1.imshow(binary_warped, cmap='gray')

    # plot histogram
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.autoscale(enable=True, axis='both', tight=True)
    plt.setp(ax1.get_xticklabels(), visible=False)
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    ax2.plot(histogram, 'blue')

    canvas.draw()       # draw the canvas, cache the renderer
    histogram_img = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    # turn 1-d array into proper shape
    width, height = fig.get_size_inches() * fig.get_dpi()
    # clear figure
    plt.clf()
    hist = histogram_img.reshape(height, width, 3)
    hist = cv2.resize(hist, (binary_warped.shape[1], binary_warped.shape[0]))
    return hist


def mask_lane_line(binary_warped, warped_img, lines, draw_rect=True, nwindows=9, margin=20, minpix=50):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = margin
    # Set minimum number of pixels found to recenter window
    minpix = minpix
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    left_line, right_line = lines
    if (left_line.use_best_fit and right_line.use_best_fit) and None not in (left_line.best_fit, right_line.best_fit):
        left_fit, right_fit = left_line.best_fit, right_line.best_fit
        left_lane_inds = (
        (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
        nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = (
        (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
        nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))
    else:
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            if draw_rect:
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their median(!) position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.median(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.median(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    left_x = nonzerox[left_lane_inds]
    left_y = nonzeroy[left_lane_inds]
    right_x = nonzerox[right_lane_inds]
    right_y = nonzeroy[right_lane_inds]

    new_y_range = (0, binary_warped.shape[0])
    if left_x.size > 0:
        left_status = left_line.add_raw_pts(left_x, left_y, new_y_range)
    if right_x.size > 0:
        right_status = right_line.add_raw_pts(right_x, right_y, new_y_range)
    return out_img


def fill_lane_line(warped_img, lines, draw_fit_line=False):
    left_line, right_line = lines
    left_fitx, left_fity = left_line.get_fitted_pts((0, warped_img.shape[0]))
    right_fitx, right_fity = right_line.get_fitted_pts((0, warped_img.shape[0]))

    left_pts = np.transpose(np.vstack([left_fitx, left_fity]))
    right_pts = np.flipud(np.transpose(np.vstack([right_fitx, right_fity])))
    pts = np.vstack([left_pts, right_pts])

    # poly fill
    lane_filled_img = cv2.fillPoly(np.copy(warped_img), np.int_([pts]), (0, 255, 0))
    if draw_fit_line:
        for ps in [left_pts, right_pts]:
            cv2.polylines(lane_filled_img, np.int_([ps]), False, (0, 0, 255), 10)
    return lane_filled_img


def unwarp_image(lane_filled_img, orig_image, trans_m_inv):
    newwarp = cv2.warpPerspective(lane_filled_img, trans_m_inv, (orig_image.shape[1], orig_image.shape[0]))
    result = cv2.addWeighted(orig_image, 1, newwarp, 0.3, 0)
    return result


def put_text(unwarp_img, lines):
    left_line, right_line = lines
    # Now our radius of curvature is in meters
    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700
    radius, midpoint = left_line.calculate_radius_and_midpoint(right_line, unwarp_img.shape[0],
                                                               ym_per_pix=ym_per_pix, xm_per_pix=xm_per_pix)
    offset = abs(unwarp_img.shape[1] // 2 - midpoint) * xm_per_pix
    text = "radius: %d m, offset: %0.2f m" % (int(radius), offset)

    cv2.putText(unwarp_img, text, (30, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    return unwarp_img


def resize_and_stack(img):
    resized = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    if len(img.shape) == 2:
        # convert binary image back to RGB image
        resized = np.dstack([resized, resized, resized]) * 255
    return resized


def uv_channel(img, t_min=-255, t_max=50):
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    uv = yuv[:, :, 1].astype(np.int32) - yuv[:, :, 2].astype(np.int32)
    uv[(uv > t_min) & (uv < t_max)] = 1
    uv[(uv <= t_min) | (uv >= t_max)] = 0
    # flip 0 and 1
    uv[:, :] = -(uv[:, :] - 1)
    return uv


def sobel_channel(test_img, threshold=(20, 100), orientation='x'):
    gray = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
    sxbinary = np.zeros_like(gray)
    for orient in [(1, 0), (0, 1)]:
        #         orient = (1, 0) if orientation is 'x' else (0, 1)
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, orient[0], orient[1]))

        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        sxbinary[(scaled_sobel >= threshold[0]) & (scaled_sobel <= threshold[1])] = 1
        # sxbinary[(scaled_sobelx >= thresh_min) & (scaled_sobelx <= thresh_max) \
        #         & (scaled_sobely >= thresh_min) & (scaled_sobely <= thresh_max)] = 1
    return sxbinary


def visualize_pipeline(files, procedures, figsize=(8, 15)):
    """
     read the image from files(paths)
     and then sequentially apply procedures to the image(s)
     each procedure accept all previous proccessed image and
     will return a result image
     finally, plot all intermediate results for each image in a subplots
    """
    num_of_files = len(files)
    plt.figure(figsize=figsize)
    for idx, f in enumerate(files):
        img = mpimg.imread(f)
        results = apply_pipeline(img, procedures=procedures, return_all=True)
        assert type(results) is list
        subplot_per_row = len(results)
        for i, image in enumerate(results):
            ax = plt.subplot(num_of_files, subplot_per_row, idx * subplot_per_row + i + 1)
            plt.title(f.split("/")[-1], fontsize=5)
            plt.setp(ax.get_xticklines(), visible=False)
            plt.setp(ax.get_yticklines(), visible=False)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            plt.imshow(image)


def apply_pipeline(image, procedures, return_all=False):
    results = [image]
    for p_idx, p in enumerate(procedures):
        # each procedure can access to all previous results
        r = p(results)
        results.append(r)
    if return_all:
        return results
    else:
        return r


def stitch_diagnostic_img(original, images):
    h = original.shape[0] // 2
    w = original.shape[1] // 2
    stitch = np.zeros((original.shape[0] * 1.5, original.shape[1], 3), dtype=np.uint8)

    img1, img2, img3, img4, img5 = images[:5]
    stitch[:h, :w, :] = resize_and_stack(original)

    stitch[:h, w:, :] = resize_and_stack(img1)

    stitch[h:2*h, :w, :] = resize_and_stack(img2)
    stitch[h:2*h, w:, :] = resize_and_stack(img3)

    stitch[2*h:, :w, :] = resize_and_stack(img4)
    stitch[2*h:, w:, :] = resize_and_stack(img5)
    return stitch


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: %s [input_path] [output_path]" % sys.argv[0])
        sys.exit(-1)

    print("start camera calibration...")
    calibration_file_pattern = "camera_cal/calibration*.jpg"
    obj_points, img_points = findChessboardCornes(calibration_file_pattern, nx=9, ny=6)


    # since all the pictures are of same size
    # we only have to call calibrateCamera once
    img_size = (1028, 720)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points,
                                                       img_size, None, None)

    src = np.array([(300, 700), (1100, 700), (550, 500), (800, 500)], np.float32)
    dst = np.array([(500, 700), (800, 700), (500, 525), (800, 525)], np.float32)
    trans_m, trans_m_inv = get_transformation_matrix(src, dst)

    # use Line instance to hold the lane line data for the current frame,
    # as well as passes data between procedurecs
    use_best_fit = True
    lines = [Line(use_best_fit=use_best_fit), Line(use_best_fit=use_best_fit)]
    thresholds = [
        (None, 0, 153, 255, 15000),  # RGB normal
        (cv2.COLOR_RGB2HLS, 2, 150, 255, 10000),  # S Channel HLS normal
        (None, 0, 200, 255, 10000),  # RGB for shadow bridge e.g. test1.jpg
        (cv2.COLOR_RGB2YUV, 1, 150, 171, 8000)  # YUV Channel U
    ]

    # an extra selector for stack_channel_binary
    extra_channel = [
        lambda img: sobel_channel(img, orientation='x', threshold=(20, 100)),
        lambda img: uv_channel(img)
    ]

    procedures = [
        lambda imgs: cv2.undistort(imgs[-1], mtx, dist, None, mtx),
        lambda imgs: cv2.warpPerspective(imgs[-1], trans_m, img_size, flags=cv2.INTER_LINEAR),
        lambda imgs: stack_channel_binary(imgs[-1], thresholds, extra_channel=extra_channel),
        # select region on bird's eye view to avoid unwanted edges
        lambda imgs: region_select(imgs[-1], (450, 700), (180, 0), (830, 700), (1000, 0), fill=0),
        lambda imgs: mask_lane_line(imgs[-1], imgs[-3], lines, draw_rect=True, nwindows=50, margin=40, minpix=50),
        lambda imgs: fill_lane_line(imgs[-4], lines, draw_fit_line=False),
        lambda imgs: unwarp_image(imgs[-1], imgs[0], trans_m_inv),
        lambda imgs: put_text(imgs[-1], lines),
        # lambda imgs: stitch_diagnostic_img(imgs[2], [imgs[3], imgs[4], imgs[5], imgs[6], imgs[7]])
    ]
    # visualize_pipeline(glob('test_images/*.jpg'),procedures=procedures, figsize=(15,8))

    print("start processing video: %s save to: %s" % tuple(sys.argv[1:]))

    clip1 = VideoFileClip(sys.argv[1])
    white_clip = clip1.fl_image(lambda img: apply_pipeline(img, procedures))
    white_clip.write_videofile(sys.argv[2], audio=False)
