import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, normalize
from skimage.feature import hog
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from sklearn.externals import joblib
# import seaborn as sns

 function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan / nx_pix_per_step) - 1
    ny_windows = np.int(yspan / ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# Define a function to compute color histogram features
# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
def color_hist(img, nbins=32):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:, :, 0], bins=nbins)
    ghist = np.histogram(img[:, :, 1], bins=nbins)
    bhist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

def get_hog_features_cv2(img):
    winSize = (64,64)
    blockSize = (16,16)
    blockStride = (2,2)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 1
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    #compute(img[, winStride[, padding[, locations]]]) -> descriptors
    winStride = (8,8)
    padding = (8,8)
    locations = ((10,20),)
    h = hog.compute(img.astype(np.uint8),winStride,padding,locations)
    return h
    

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', orient=9, spatial_size=(32, 32),
                     hist_bins=32, hist_feat=True, spatial_feat=True, hog_feat=True,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        fs = []
        # Read in each one by one
        image = mpimg.imread(file)
        if file.endswith("jpg"):
            image = image.astype(np.float32) / 255
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        # normalize feature_image, due to varies reasons,
        # the converted img have different value range
        for c in range(3):
            feature_image[:, :, c] = feature_image[:, :, c] / np.max(feature_image[:, :, c])

        if spatial_feat:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            fs.append(spatial_features)
        # Apply color_hist() also with a color space option now
        if hist_feat:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            fs.append(hist_features)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_feat:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)

#             hog_features = np.ravel(get_hog_features_cv2(feature_image))
#             print(fs.shape)
            fs.append(hog_features)


        # Append the new feature vector to the features list
        features.append(np.concatenate(fs))
        # delete image instance to save memory
        del image
    # Return list of feature vectors
    return features


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(image, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) Define an empty list to receive features
    fs = []
    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(image)

    # normalize feature_image, due to varies reasons,
    # the converted img have different value range
    for c in range(3):
        feature_image[:, :, c] = feature_image[:, :, c] / np.max(feature_image[:, :, c])

    if spatial_feat:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        fs.append(spatial_features)
    # Apply color_hist() also with a color space option now
    if hist_feat:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        fs.append(hist_features)

    # Call get_hog_features() with vis=False, feature_vec=True
    if hog_feat:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
#         hog_features = np.ravel(get_hog_features_cv2(feature_image))
        fs.append(hog_features)

    # Append the new feature vector to the features list
    return np.concatenate(fs)


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    # 1) Create an empty list to receive features
    all_features = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) append features to a list
        all_features.append(features)
    all_features = np.nan_to_num(np.array(all_features))
    scaled_all_features = scaler.transform(all_features)
    # 6) predict all images in a single batch
    predictions = svc.predict(scaled_all_features)
    on_windows = np.array(windows)[predictions == 1]
    return on_windows


# Read in a pickle file with bboxes saved
# Each item in the "all_bboxes" list will contain a
# list of boxes for one of the images shown above
# Read in image similar to one shown above


def add_heat(heatmap, bbox_list, heat_value=1):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += heat_value

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # check the w:h ratio, reject outlier
        height = bbox[1][1] - bbox[0][1]
        width = bbox[1][0] - bbox[0][0]
        if height / width >= 1.5 or width < 64 or height < 64:
            continue
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
        cv2.putText(img, "%dx%d" % (width, height), (bbox[0][0] + width // 2, bbox[0][1] - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)
    # Return the image
    return img


def make_heatmap(image, box_list, threshold=1):
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, box_list)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, threshold)
    # # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    return heatmap


def process_frame(img, svc, windows, hot_windows_buffer, cnt, X_scaler, img_to_draw=None, hyper=hyperparameter, skip=1, buffer_size=5, min_threshold=20):
    cnt[0] += 1
    original_img = img
    img_to_draw = img_to_draw if img_to_draw is not None else original_img
    if cnt[0] % skip == 0:
        original_img = np.copy(img)
        # scale the img to [0, 1]
        img = img.astype(np.float32) / 255
        hot_windows = search_windows(img, windows, svc, X_scaler, color_space=hyperparameter['colorspace'],
                                     hog_channel=hyper['hog_channel'], spatial_size=hyper['spatial_size'],
                                     hist_bins=hyper['hist_bins'], hist_range=hyper['hist_range'],
                                     orient=hyper['orient'], pix_per_cell=hyper['pix_per_cell'],
                                     cell_per_block=hyper['cell_per_block'],
                                     spatial_feat=hyper['spatial_feat'], hist_feat=hyper['hist_feat'],
                                     hog_feat=hyper['hog_feat'])

        print('hot windows length: ', len(hot_windows))
        if buffer_size > 0 and len(hot_windows_buffer) >= buffer_size:
            hot_windows_buffer.pop(0)

        hot_windows_buffer.append(hot_windows)
    elif len(hot_windows_buffer) == 0:
        return img_to_draw
    else:
        hot_windows = hot_windows_buffer[-1]

    all_windows = []
    for ws in hot_windows_buffer:
        all_windows.extend(ws)
    threshold = max(int(len(all_windows) * .15), min_threshold)

    heatmap = make_heatmap(img_to_draw, all_windows, threshold=threshold)

    # # Find final boxes from heatmap using label function
#     labels = label(heatmap)
#     boxed_img = draw_labeled_bboxes(original_img, labels)
    
#     return boxed_img
    return heatmap


def get_features(cars, notcars, X_scaler, spatial_feat, hist_feat, hog_feat, colorspace,
                     orient, spatial_size, pix_per_cell, cell_per_block, hog_channel,
                     hist_bins):
    t = time.time()
    car_features = extract_features(cars, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat,
                                    cspace=colorspace, orient=orient, spatial_size=spatial_size,
                                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, hist_bins=hist_bins)
    notcar_features = extract_features(notcars, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat,
                                       cspace=colorspace, orient=orient, spatial_size=spatial_size,
                                       pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, hist_bins=hist_bins)

    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to extract HOG features...')
    # Create an array stack of feature vectors
    print(car_features[0].shape, notcar_features[0].shape, len(car_features), len(notcar_features))
    X = np.vstack([car_features, notcar_features]).astype(np.float64)

    # encountered error: _hog.py:88: RuntimeWarning: invalid value encountered in sqrt
    #  image = np.sqrt(image)
    # use this function to remove nan
    X = np.nan_to_num(X)

    # Fit a per-column scaler
    X_scaler.fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)


    # scaled_X = normalize(X, norm='max')
    # feature_indices = np.arange(scaled_X.shape[1])
    # feature_indices = (feature_indices != 784) & (feature_indices != 800) & (feature_indices != 768)
    # # remove feature that is too big. don't know how to normalized it
    # scaled_X = scaled_X[:, feature_indices]


    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    
    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(scaled_X[0]))
    return scaled_X, y


def train_classifier(scaled_X, y):

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)


    # Use a linear SVC
    svc = LinearSVC(dual=False)
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    score = round(svc.score(X_test, y_test), 4)
    print('Test Accuracy of SVC = ', score)
    # Check the prediction time for a single sample
    t = time.time()
    #     n_predict = 10
    #     print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    #     print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    #     t2 = time.time()
    #     print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
    return svc, score


def load_and_extract(sample_size, X_scaler, seed=None, hyper=hyperparameter):
    print(hyperparameter)
    notcars_files      = glob.glob("train_data/non-vehicles/*/*.png")
#     cars_files = glob.glob("train_data/vehicles/*/*.png")                      # size = 
    cars_files         = glob.glob("train_data/vehicles/*/*.png")  #  size=8800))
    autti_cars_files   = glob.glob("udacity_data/autti/car1/*.jpg") # size=5000)
    crowdai_cars_files = glob.glob("udacity_data/crowdai/car_img_resized/*.jpg") #, size=22000)
    cars_files.extend(autti_cars_files)
#     cars_files.extend(crowdai_cars_files)

    if seed is not None:
        np.random.seed(seed)
    cars = np.random.choice(cars_files, size=sample_size)
    notcars = np.random.choice(notcars_files, size=sample_size)


    return  get_features(cars, notcars, X_scaler, hyper['spatial_feat'], hyper['hist_feat'], hyper['hog_feat'],
                                            hyper['colorspace'], hyper['orient'], hyper['spatial_size'],
                                            hyper['pix_per_cell'], hyper['cell_per_block'], hyper['hog_channel'],
                                            hyper['hist_bins'])


def detect_vehicle(svc, X_scaler, hyper, video_file='test_video.mp4', output='boxed_test_video.mp4'):
    all_windows = []
    image_shape = (720, 1280, 3)
    for window_size, y_start_stop in zip([96, 112, 128], [[350, 450], [380, 480], [400, 500]]):
        windows = slide_window(np.ones(image_shape), y_start_stop=y_start_stop, xy_window=(window_size, int(window_size * 0.85)),
                               xy_overlap=(.8,.8))
        all_windows.extend(windows)

    hot_windows_buffer = []
    cnt = [0]
    clip1 = VideoFileClip(video_file)
    white_clip = clip1.fl_image(lambda img: process_frame(img, svc, all_windows, hot_windows_buffer, cnt, X_scaler,
                                                          hyper=hyper, skip=5, buffer_size=4, min_threshold=10))
    white_clip.write_videofile(output, audio=False)


if __name__ == '__main__':

    hyperparameter = dict(\
	colorspace = 'YCrCb',  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
	orient = 9,
	pix_per_cell = 8,
	cell_per_block = 4,
	hog_channel = 'ALL',  # Can be 0, 1, 2, or "ALL"
	spatial_size = (32, 32),
	hist_bins = 32,
	spatial_feat = True,
	hist_feat = False,
	hist_range = (0, 256),
	hog_feat = True)

    all_windowp = []
    image_shape = (720, 1280, 3)
    for window_size, y_start_stop in zip([96, 112, 128], [[350, 450], [380, 480], [400, 500]]):
	windows = slide_window(np.ones(image_shape), y_start_stop=y_start_stop, xy_window=(window_size, int(window_size * 0.85)),
			       xy_overlap=(.8,.8))
	all_windows.extend(windows)

    hot_windows_buffer = []
    cnt = [0]
    # detect_vehicle = lambda img: process_frame(img, svc, all_windows, hot_windows_buffer, cnt, X_scaler,
#                     hyper=hyper, skip=5, buffer_size=4, min_threshold=10)


    X_scaler = StandardScaler()
    X, y = load_and_extract(35000, X_scaler, seed=None)
    svc, score = train_classifier(X, y)


    detect_vehicle(svc, X_scaler, hyper=hyperparameter, video_file='project_video.mp4', output='bboxed_project_video.mp4')                     
                     
