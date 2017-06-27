# Vehicle Detection and Lane Finding


## Final Video
[Lane finding and vehicle detection project video](https://youtu.be/Qjrnx3H-1tc)

## Vehicle Detection
### Training Dataset:
1. KIT
2. autti
3. crowd.ai

A total number 35k car image and 35k non car images were used in training, 20% of them were used in validation phase.

### Feature Selection
The images were transformed into [color space]
Histogram Orientation Gradients
	HOG feature extractor implemented in scikit-image
	orient = 9, pix_per_cell = 8, cell_per_block = 4, hog_channel = 'ALL',

histogram

### Classifier
Linear SVM Classifier

Training error:
Validation error:

### Detection

#### sliding window
square shaped sliding window of size 96, 112, 128 pixels were used to search interested area of the video. buffer the window position for positive predictions.

#### Skip frames
To increase processing speed, only 1 in every 5 frames are processed.

#### Eliminate False Positives
To eliminate false positives, buffering and thresholding were used:
* buffering prediction result for last five frames, apply a dynamic threshold to pixels that have positive predictions, the threshold value is determined by the total number of positive windows in the buffer.
[heatmap]
[threshold]


## Lane Finding

To mark the lane in which the vehicle is driving, following processing pipelines were implemented.

#### Camera Calibration

Using chessboard, the images have been un distorted to adjust the 

#### unDistort image
#### Birds eye view
Using perspective transform to turn the image into Birds eye view, so

![Bird's eye view of the road](test_images/birdseye.png)

#### Detect lane edges

By transform the image to different color spaces and apply different threshold ranges, stacking the resulted binary images, the lane edges could be separated from background and other objects reasonably 
* R of RGB range from 153, 255 
* R of RGB range from 200, 255
* S of HLS range from 150 255
* U of YUV range from 150 171
* Use Sobel operator to detect edges in the interested area, since lane lines directions are mostly vertical in the image, so sobel operator was applied along x-axe.

Because variation of lighting, shadowing and other interference. the accuracy of different color space and threshold ranges varies. To reduce false positives, another threshold was applied to total number of positive pixels, if the threshold is exceeded the result won't be merged into final output.

![Edge detection](test_images/binary.png)


#### Histogram
A histogram of non-zero pixels of calculated, two peaks of the histogram are seen as the most like position of lane line edges

![Histogram](test_images/histogram.png)

#### Probing Window
apply two series of probing windows from bottom to top, to locate all the non-zero pixels in lane edge area. As the windows progressing along y-axe, its center was repeated recalculated to allow the window follows the bending or leaning lane edge.

![Probing Window](test_images/probing_window.png)

#### Polyfit
Fit the coordinates of the non-zero pixels to a second order polynomial.
Reasoning that the fitted line won't change dramatically, the program will reject fitting result that changes too much.

![Fit and fill the lane](test_images/fill.png)

