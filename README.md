# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car.png "car"
[image2]: ./output_images/notcar.png "not car"
[image3]: ./output_images/hogcar.png "hog car"
[image4]: ./output_images/hognotcar.png "hog not car"

[image5]: ./output_images/grid0-4.png "grid-4"
[image6]: ./output_images/grid0-3.png "grid-3"
[image7]: ./output_images/grid0-2.png "grid-2"
[image8]: ./output_images/grid0-1.png "grid-1"

[image9]: ./output_images/pipeline.png "Boxes"

[image10]: ./output_images/4.png "Pipeline 1"
[image11]: ./output_images/3.png "Pipeline 2"
[image12]: ./output_images/2.png "Pipeline 3"
[image13]: ./output_images/1.png "Pipeline 4"

[image14]: ./output_images/heat1.png "Heat 1"
[image15]: ./output_images/heat2.png "Heat 2"
[image16]: ./output_images/heat3.png "Heat 3"
[image17]: ./output_images/heat4.png "Heat 4"
[image18]: ./output_images/heat5.png "Heat 5"
[image19]: ./output_images/heat6.png "Heat 6"

[image20]: ./output_images/labels6.png "Labels"
[image21]: ./output_images/box6.png "Box"

[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in './project.py' (lines #33 through #72 in the function 'prepare_data').

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

Car
![alt text][image1]
Not Car
![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `LUV` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

Car
![alt text][image3]
Not Car
![alt text][image4]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found that the classifier performed best using the following parameters:

- color_space = 'LUV'
- orient = 11
- pix_per_cell = 8
- cell_per_block = 2
- hog_channel = 'ALL'
- spatial_size = (16, 16)
- hist_bins = 32

I based the performance of the parameters on the accuracy of classifying images in the test set, which was about 98 percent. I also continually checked against the test images and video.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG features (using all color channels) as well as a histogram of color channels and a spatial function with size 16x16. The code related to the classifier can be found in './lib/classifier.py'. The code for extracting features can be found in './lib/features.py'.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I tried many different parameters for the sliding window search. I used the find_cars function from the lessons to perform the search, but I ran it in many different sections and scales on the image.

I used smaller scales toward the middle of the image where cars would typically be seen at smaller sizes and I used larger scales toward the bottom of the image where cars would be larger. Also, the smaller the scale, the less area was needed to search horizontally, since the road narrows in the distance.

I used 4 different scales (0.75, 1, 1.4, 1.8) which really seemed to help when trying to eliminate false positives.

Here are some examples of the windows/scales searched:

![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 4 scales using YUV 3-channel HOG, which provided a nice result. I tried many different parameters for the feature extraction and HOG function, but found these parameters worked best with my pipeline. In order to increase the performance of the classifier I used the decision_function to adjust the activation threshold. This helped to eliminate false positives. I also adjusted the "C" value to 0.001 to reduce overfitting.

Here is an example image:

![alt text][image9]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

To further eliminate false positives, I stored the car detections from the previous 9 frames. Then for each new frame I used the current & previous frame data to construct the heatmap. That way I could use a greater threshold for the heatmap because if a previous detection was a false positive, it is typically less likely for that detection to occur in multiple frames.

### Here are six frames and their corresponding heatmaps:

![alt text][image14]
![alt text][image15]
![alt text][image16]
![alt text][image17]
![alt text][image18]
![alt text][image19]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image20]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image21]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I found it to be difficult to find the appropriate method to extract features for the classifier. I tried many different parameters for the HOG function and eventually seemed to find something that worked, but it did seem to give a lot of false positives. I then noticed that I was not converting the frames of my video to BGR like the pipeline expected. Once I did this conversion my results seemed to get much better.

I found it to be a challenge to remove those false positives from the video. My method of saving X number of previous frames and using those in the heatmap seemed to help a lot, but it isn't fool proof.

I also found that the detection of white vehicles sometimes seemed problematic. I think it would help to find more data of white vehicles to use in the training set. It might also help to use different color maps and color features.
