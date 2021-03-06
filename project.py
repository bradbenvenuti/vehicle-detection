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
from collections import deque
from sklearn.externals import joblib
from lib.windows import *
from lib.features import *
from lib.classifier import *
from moviepy.editor import VideoFileClip

def draw_grid(img, ystart, ystop, xstart, xstop, scale, pix_per_cell, cell_per_block):

	draw_img = np.copy(img)

	img_tosearch = img[ystart:ystop,xstart:xstop,:]
	if scale != 1:
		imshape = img_tosearch.shape
		img_tosearch = cv2.resize(img_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

	ch1 = img_tosearch[:,:,0]
	ch2 = img_tosearch[:,:,1]
	ch3 = img_tosearch[:,:,2]

	# Define blocks and steps as above
	nxblocks = (ch1.shape[1] // pix_per_cell) + 1
	nyblocks = (ch1.shape[0] // pix_per_cell) + 1

	# 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
	window = 64
	nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
	cells_per_step = 2  # Instead of overlap, define how many cells to step
	nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
	nysteps = (nyblocks - nblocks_per_window) // cells_per_step

	color = (randint(0,255),randint(0,255),randint(0,255))
	for xb in range(nxsteps):
		for yb in range(nysteps):
			ypos = yb*cells_per_step
			xpos = xb*cells_per_step

			xleft = xpos*pix_per_cell
			ytop = ypos*pix_per_cell

			# Extract the image patch
			subimg = cv2.resize(img_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

			xbox_left = np.int(xleft*scale) + xstart
			ytop_draw = np.int(ytop*scale)
			win_draw = np.int(window*scale)
			# box_list.append([[xbox_left, ytop_draw+ystart],[xbox_left+win_draw,ytop_draw+win_draw+ystart]])
			cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),color,2)

	return draw_img

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

	mpimg.imsave('./output_images/car.png', vehicles[50])
	mpimg.imsave('./output_images/notcar.png', non_vehicles[50])
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

	# hog_features_car, hog_features_car_img = get_hog_features(vehicles[50][:,:,0], orient,
	# 		pix_per_cell, cell_per_block, vis=True, feature_vec=True)
	# hog_features_not_car, hog_features_not_car_img = get_hog_features(non_vehicles[50][:,:,0], orient,
	# 		pix_per_cell, cell_per_block, vis=True, feature_vec=True)
	# mpimg.imsave('./output_images/hogcar.png', hog_features_car_img)
	# mpimg.imsave('./output_images/hognotcar.png', hog_features_not_car_img)

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
orient = 11
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True
prev_boxes = deque(maxlen = 9)

# Setup classifier
X_train, y_train, X_test, y_test, X_scaler = prepare_data(color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
														cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
classifier = fit_classifier(X_train, y_train)
# test_classifier(classifier, X_test, y_test)

# Process Frames
def process_image(img, single = False):
	# draw_img = np.copy(img)
	global prev_boxes

	box_list = []
	box_list += find_cars(img, 400, 592, 128, 1152, 0.75, classifier, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space, 1.0)
	box_list += find_cars(img, 400, 592, 48, 1232, 1.0, classifier, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space, 1.0)
	box_list += find_cars(img, 384, 656, 16, 1280, 1.4, classifier, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space, 1.0)
	box_list += find_cars(img, 384, 656, 16, 1280, 1.8, classifier, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space, 1.0)
	# for box in box_list:
	# 	cv2.rectangle(draw_img,(box[0][0], box[0][1]),(box[1][0],box[1][1]),(0,0,255),6)

	# print(box_list)
	curr_boxes = list(box_list)
	if (single == False):
		for prev_frame in prev_boxes:
			for boxes in prev_frame:
				curr_boxes.append(boxes)

	prev_boxes.append(box_list)

	heatm = np.zeros_like(img[:,:,0]).astype(np.float)
	heatm = add_heat(heatm, curr_boxes)
	if (single == True):
		heat = apply_threshold(heatm, 1)
	else:
		heat = apply_threshold(heatm, 4)
	# Visualize the heatmap when displaying
	heatmap = np.clip(heat, 0, 255)
	# Find final boxes from heatmap using label function
	labels = label(heatmap)

	draw_img = draw_labeled_bboxes(np.copy(img), labels)
	return draw_img

def process_video(img):
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	img = process_image(img)
	return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # Run on test images
# imgs = fetch_images('./test_images/test1.png')
# num = 0
# for img in imgs:
# 	draw_img = process_image(img, True)
# 	draw_img = draw_grid(img, 400, 592, 128, 1152, 0.75, pix_per_cell, cell_per_block)
# 	mpimg.imsave('./output_images/grid0-1.png', draw_img)
# 	draw_img = draw_grid(img, 400, 592, 48, 1232, 1.0, pix_per_cell, cell_per_block)
# 	mpimg.imsave('./output_images/grid0-2.png', draw_img)
# 	draw_img = draw_grid(img, 384, 656, 16, 1280, 1.4, pix_per_cell, cell_per_block)
# 	mpimg.imsave('./output_images/grid0-3.png', draw_img)
# 	draw_img = draw_grid(img, 384, 656, 16, 1280, 1.8, pix_per_cell, cell_per_block)
# 	mpimg.imsave('./output_images/grid0-4.png', draw_img)
# 	num = num + 1

prev_boxes.clear()
output = 'out.mp4'
clip1 = VideoFileClip("project_video.mp4")
clip = clip1.fl_image(process_video)
%time clip.write_videofile(output, audio=False)
