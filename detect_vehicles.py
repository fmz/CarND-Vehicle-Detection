import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import cv2
import glob
import time
import pickle
from helper_functions import *
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
        cell_per_block, spatial_size, hist_bins, spatial_feat, color_space, hist_feat,
        hog_feat, out_heatmap):

    draw_img = np.copy(img)
    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]
    # apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else: ctrans_tosearch = np.copy(img_tosearch)

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), \
            np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    box_list = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            features = []
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Get color features
            if (spatial_feat == True) or (hist_feat == True):

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
                if spatial_feat == True:
                    features.append(bin_spatial(subimg, size=spatial_size))
                if hist_feat == True:
                    features.append(color_hist(subimg, nbins=hist_bins))

            # Extract HOG for this patch
            if hog_feat == True:
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                features.append(np.hstack((hog_feat1, hog_feat2, hog_feat3)))

            # Scale features and make a prediction
            features = np.concatenate(features).astype(np.float64).reshape(1,-1)
            test_features = X_scaler.transform(features)
            test_prediction = svc.predict(test_features)

            # If we found a car, keep track of the binding box.
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                box_list.append (((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+ \
                    win_draw+ystart)))
                #cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+ \
                #    win_draw+ystart),(0,0,255),6)


    heat = np.zeros_like(draw_img[:,:,0]).astype(np.float)
    heat = add_heat(heat, box_list)
    heat = apply_threshold(heat, 2)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(draw_img, labels)

    # Just for the writeup
    if out_heatmap:
        mpimg.imsave("heatmap_example.jpg", heatmap, cmap='hot')
        mpimg.imsave("boxes_example.jpg", draw_img)

    return draw_img

# The following 3 lines are needed because moviepy.fl_image takes a function that takes only an image.
detector = []
def _find_cars(image):
    global detector
    return detector.call_find_cars(image, False)

class car_detector:
    def __init__(self, cars, notcars, do_update=True, pickle_file = ''):
        self.y_start = 350 # Top portion of the region of interest.
        self.y_stop = 656 # Bottom portion of the region of interest.
        self.scale = 1.55

        if (do_update or pickle_file == ''):
            self.color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
            self.orient = 13  # HOG orientations
            self.pix_per_cell = 8 # HOG pixels per cell
            self.cell_per_block = 2 # HOG cells per block
            self.hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
            self.spatial_size = (16, 16) # Spatial binning dimensions
            self.hist_bins = 32  # Number of histogram bins
            self.spatial_feat = True # Spatial features on or off
            self.hist_feat = True # Histogram features on or off
            self.hog_feat = True # HOG features on or off

            car_features = extract_features(cars, color_space=self.color_space,
                                spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                orient=self.orient, pix_per_cell=self.pix_per_cell,
                                cell_per_block=self.cell_per_block,
                                hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                hist_feat=self.hist_feat, hog_feat=self.hog_feat)

            notcar_features = extract_features(notcars, color_space=self.color_space,
                                spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                orient=self.orient, pix_per_cell=self.pix_per_cell,
                                cell_per_block=self.cell_per_block,
                                hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                hist_feat=self.hist_feat, hog_feat=self.hog_feat)

            X = np.vstack((car_features, notcar_features)).astype(np.float64)
            # Fit a per-column scaler
            self.X_scaler = StandardScaler().fit(X)
            # Apply the scaler to X
            scaled_X = self.X_scaler.transform(X)

            # Define the labels vector
            y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

            # Split up data into randomized training and test sets
            rand_state = np.random.randint(0, 100)
            X_train, X_test, y_train, y_test = train_test_split(
                scaled_X, y, test_size=0.2, random_state=rand_state)

            print('Using:',self.orient,'orientations',self.pix_per_cell,
                'pixels per cell and', self.cell_per_block,'cells per block')
            print('Feature vector length:', len(X_train[0]))
            # Use a linear SVC
            self.svc = LinearSVC()
            #self.svc = SVC()
            # Check the training time for the SVC
            t=time.time()
            self.svc.fit(X_train, y_train)
            t2 = time.time()
            print(round(t2-t, 2), 'Seconds to train SVC...')
            # Check the score of the SVC
            print('Test Accuracy of SVC = ', round(self.svc.score(X_test, y_test), 4))

            if pickle_file != '':
                pickle_dict = { "color_space": self.color_space,
                                "orient": self.orient,
                                "pix_per_cell": self.pix_per_cell,
                                "cell_per_block": self.cell_per_block,
                                "hog_channel": self.hog_channel,
                                "spatial_size": self.spatial_size,
                                "hist_bins": self.hist_bins,
                                "spatial_feat": self.spatial_feat,
                                "hist_feat": self.hist_feat,
                                "hog_feat": self.hog_feat,
                                "svc": self.svc,
                                "X_scaler": self.X_scaler
                              }
                pickle_out = open(pickle_file, "wb")
                pickle.dump(pickle_dict, pickle_out)

        else:
            pickle_in = pickle.load(open(pickle_file, "rb"))
            self.color_space = pickle_in["color_space"]
            self.orient = pickle_in["orient"]
            self.pix_per_cell = pickle_in["pix_per_cell"]
            self.cell_per_block = pickle_in["cell_per_block"]
            self.hog_channel = pickle_in["hog_channel"]
            self.spatial_size = pickle_in["spatial_size"]
            self.hist_bins = pickle_in["hist_bins"]
            self.spatial_feat = pickle_in["spatial_feat"]
            self.hist_feat = pickle_in["hist_feat"]
            self.hog_feat = pickle_in["hog_feat"]
            self.svc = pickle_in["svc"]
            self.X_scaler = pickle_in["X_scaler"]


    def call_find_cars(self, image, out_heatmap):
            return find_cars(image, self.y_start, self.y_stop, self.scale, self.svc, self.X_scaler,
                            self.orient, self.pix_per_cell, self.cell_per_block, self.spatial_size,
                            self.hist_bins, self.spatial_feat, self.color_space, self.hist_feat,
                            self.hog_feat, out_heatmap)

    def detect_image(self, test_images, to_screen=True, outdir=''):
        out_images = []

        for image_name in test_images:
            image = mpimg.imread(image_name)
            out_images.append(self.call_find_cars(image, True))

        if to_screen:
            grid = gridspec.GridSpec(3, 3)
            plt.figure(figsize=(12,12))
            for i in range(len(out_images)):
                subp = plt.subplot(grid[i])
                subp.set_aspect('equal')

                img = out_images[i]
                plt.subplot(3,3,i+1)
                plt.imshow(img)
                plt.axis('off')
                plt.title(str(i))

            plt.show()

        if outdir != '':
            for i in range(len(out_images)):
                img = out_images[i]
                mpimg.imsave(outdir + '/' + str(i) + '.jpg', img)

    def detect_video(self, video_name, out_file):
        clip = VideoFileClip(video_name)
        white_clip = clip.fl_image(_find_cars)
        white_clip.write_videofile(out_file, audio=False)


if __name__ == "__main__":
    global detector

    # Read in cars and notcars (I'll be hardcoding directory names throughout)
    cars = glob.glob('vehicles/*.png')
    notcars = glob.glob('non_vehicles/*.png')

    test_images = glob.glob('test_images/*.jpg')

    detector = car_detector(cars, notcars, True, 'model.p')
    detector.detect_image(test_images=test_images, outdir="output_images")
    detector.detect_video("project_video.mp4", "project_video_out.mp4")

