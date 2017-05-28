%matplotlib inline
%config InlineBackend.figure_format = 'svg'
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import glob
import time
from sklearn.externals import joblib
from lib.windows import *
from lib.features import *
from lib.classifier import *
from moviepy.editor import VideoFileClip

def fetch_images(path):
	imgs = []
	for file in glob.glob(path):
		imgs.append(cv2.imread(file))
	return np.array(imgs)

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    draw_img = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return draw_img

def prepare_data(color_space, spatial_size, hist_bins, orient, pix_per_cell,
                        cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat):

	# Get Images
	vehicles = fetch_images('./vehicles/*/*.png')
	non_vehicles = fetch_images('./non-vehicles/*/*.png')

	# Get Features
	t=time.time()
	vehicle_features = extract_features(vehicles, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)
	non_vehicle_features = extract_features(non_vehicles, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)

	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to train extract features...')

	# Create an array stack of feature vectors
	X = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)
	# Fit a per-column scaler
	X_scaler = StandardScaler().fit(X)
	# Apply the scaler to X
	scaled_X = X_scaler.transform(X)
	# Define the labels vector
	y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))

	# Split up data into randomized training and test sets
	rand_state = np.random.randint(0, 100)
	X_train, X_test, y_train, y_test = train_test_split(
	    scaled_X, y, test_size=0.2, random_state=rand_state)

	return X_train, y_train, X_test, y_test, X_scaler

# Configuration
color_space = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True
y_start_stop = [350, 656]
scale = 1

# Setup classifier
X_train, y_train, X_test, y_test, X_scaler = prepare_data(color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
														cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
classifier = fit_classifier(X_train, y_train)

# joblib.dump(classifier, 'classifier.pkl')
# joblib.dump(X_train, 'X_train.pkl')
# joblib.dump(y_train, 'y_train.pkl')
# joblib.dump(X_test, 'X_test.pkl')
# joblib.dump(y_test, 'y_test.pkl')
# joblib.dump(X_scaler, 'X_scaler.pkl')

# classifier = joblib.load('classifier.pkl')
# X_train = joblib.load(X_train, 'X_train.pkl')
# y_train = joblib.load(y_train, 'y_train.pkl')
# X_test = joblib.load(X_test, 'X_test.pkl')
# y_test = joblib.load(y_test, 'y_test.pkl')
# X_scaler = joblib.load(X_scaler, 'X_scaler.pkl')

# Run on test images
imgs = fetch_images('./test_images/*.png')
for img in imgs:
	draw_img = np.copy(img)
	box_list = []
	for i in range(1, 4):
		boxes = find_cars(img, y_start_stop[0], y_start_stop[1], i * scale, classifier, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
		box_list = box_list + boxes
	for box in box_list:
		cv2.rectangle(draw_img,(box[0][0], box[0][1]),(box[1][0],box[1][1]),(0,0,255),6)

	heat = np.zeros_like(img[:,:,0]).astype(np.float)
	heat = add_heat(heat, box_list)
	heat = apply_threshold(heat,1)
	# Visualize the heatmap when displaying
	heatmap = np.clip(heat, 0, 255)
	# Find final boxes from heatmap using label function
	labels = label(heatmap)
	draw_img = draw_labeled_bboxes(np.copy(img), labels)
	plt.figure()
	plt.imshow(draw_img)

def process_image(img):
	draw_img = np.copy(img)
	box_list = []
	for i in range(1, 4):
		boxes = find_cars(img, y_start_stop[0], y_start_stop[1], i * scale, classifier, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
		box_list = box_list + boxes
	for box in box_list:
		cv2.rectangle(draw_img,(box[0][0], box[0][1]),(box[1][0],box[1][1]),(0,0,255),6)

	heat = np.zeros_like(img[:,:,0]).astype(np.float)
	heat = add_heat(heat, box_list)
	heat = apply_threshold(heat,1)
	# Visualize the heatmap when displaying
	heatmap = np.clip(heat, 0, 255)
	# Find final boxes from heatmap using label function
	labels = label(heatmap)
	draw_img = draw_labeled_bboxes(np.copy(img), labels)
	return draw_img


# output = 'out.mp4'
# clip1 = VideoFileClip("project_video.mp4")
# clip = clip1.fl_image(process_image)
# %time clip.write_videofile(output, audio=False)
