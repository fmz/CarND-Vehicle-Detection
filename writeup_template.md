## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car]: ./examples/car.png
[notcar]: ./examples/not_car.png
[hog]: ./examples/HOG_example.jpg
[test_img_grid]: ./examples/test_img_grid.jpg
[heatmap]: ./examples/heatmap.jpg
[bboxes_and_heat]: ./examples/bboxes_and_heat.png
[heatmap_label]: ./examples/labels_map.png
[label_output]: ./examples/output_bboxes.png
[video]: ./project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
Note that most of the code is just copied verbatim from the lectures, which makes reading this a moot endeavor.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this is implemented in helper_functions.py:get_hog_features

I started by reading in all the `vehicle` and `non-vehicle` images. The images were all extracted as .png files in a separate directory for each class, with no sub-directories.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

Car            		|  Not Car
:------------------:|:-------------------------:
![car]  				|  ![notcar]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=13`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

Note that I also added in color features, like a color-channel histogram, and the raw pixels flattened to represent spatial features.

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters settled on the ones that gave me the best results. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM SkLearn LinearSVC, which I do, occasionally, in the car_detector constructor in detect_vehicles.py

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In detect_vehicles.py:find_cars I scan the image from the top down, and from left to the right, in boxes that overlap for each pixel. Each search window is 64 X 64 pixels, and its features are extracted in the same way training-image features get extracted. However, HOG features are extracted only once for the whole image, and only the relevant pixels get selected in the feature vector of any given window.

To make the process somewhat faster, the search is confined to y values of 350 to 656.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][hog]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from the test images, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on one of them:

### Here are six frames and their corresponding heatmaps:

![alt text][bboxes_and_heat]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][heatmap_label]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][label_output]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The first thing to note is that this technique is by no means realtime, but we can blame that on the fact that the developers of SciKit seem to find little value in adding GPU support, with highly questionable statements on their faq page like "Outside of neural networks, GPUs don’t play a large role in machine learning today."
Running this pipeline on the project video took about 25 minutes on a Core-i7 machine.

Also, this pipeline is also far from optimized. It treats every image as a separate entity that exists in a vaccum, and keeps no track of how each detection moves from frame to frame. 



