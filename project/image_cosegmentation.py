import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.segmentation import slic
from sklearn import svm
from scipy.io import loadmat

IMAGES_DIR = os.getcwd() + "/Data/Images/"
GROUND_TRUTH_DIR = os.getcwd() + "/Data/GroundTruth/"
SALIENCY = loadmat('saliency.mat')['saliency']
print "Saliency Loaded"

def getImageSets(directory):
	sets = sorted(os.listdir(directory))
	image_sets = {}
	for i in xrange(len(sets)):
		image_set_dir = directory + "/" + sets[i]
		images = sorted(os.listdir(image_set_dir))
		image_set = {'Name': sets[i], 'Images': images}
		image_sets[i] = image_set
	return image_sets

def readImages(directory, image_set, gt = False):
	set_dir = directory + image_set['Name']
	if gt == False:
		image_path = [set_dir + '/' + x for x in image_set['Images']]
	else:
		image_path = [set_dir + '/' + x.split('.')[0] + '.png' for x in image_set['Images']]
	return [cv2.imread(x) for x in image_path]

def convertToSkimage(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def getFeatures(image, n_segments = 200, compactness = 50):
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

	segments = slic(image_rgb, n_segments = n_segments, convert2lab = True, compactness = compactness)
	
	image_features = []
	for (i, seg) in enumerate(np.unique(segments)):
		h, w = image.shape[:2]
		mask = np.zeros((h,w), dtype = np.uint8)
		mask[segments == seg] = 1

		rgb = image_rgb*mask[:,:,np.newaxis]
		avg_rgb = np.mean(np.mean(rgb, axis = 0), axis = 0)/255

		lab = image_lab*mask[:,:,np.newaxis]
		avg_lab = np.mean(np.mean(lab, axis = 0), axis = 0)/255

		lab_hist = cv2.calcHist([image_lab], [0,1,2], mask, [8, 16, 16], [0, 255, 0, 255, 0, 255])
		cv2.normalize(lab_hist, lab_hist, 1.0, 0.0, cv2.NORM_L1)
		lab_hist = lab_hist.flatten()
		# lab_hist = lab_hist/np.sum(lab_hist)

		hue_hist = cv2.calcHist([image_hsv], [0], mask, [8], [0,255])
		cv2.normalize(hue_hist, hue_hist, 1.0, 0.0, cv2.NORM_L1)
		hue_hist = hue_hist.flatten()
		# hue_hist = hue_hist/np.sum(hue_hist)

		sat_hist = cv2.calcHist([image_hsv], [1], mask, [8], [0,255])
		cv2.normalize(sat_hist, sat_hist, 1.0, 0.0, cv2.NORM_L1)
		sat_hist = sat_hist.flatten()
		# sat_hist = sat_hist/np.sum(sat_hist)

		moments = cv2.moments(mask)
		cx = int(moments['m10']/moments['m00'])
		cy = int(moments['m01']/moments['m00'])
		center = np.array([cx/w, cy/h])

		x, y, width, height = cv2.boundingRect(mask)
		bbox = np.array([float(x)/w, float(y)/h, float(width)/w, float(height)/h])

		aspect_ratio = np.array([float(width)/height])

		area = np.array([np.sum(mask == 1)/(w*h)])

		features = np.concatenate((avg_rgb, avg_lab, lab_hist, hue_hist, sat_hist, center, bbox, aspect_ratio, area))
		image_features.append(features)
	
	return {'features':np.vstack(image_features), 'superpixels':segments}

def computeKernel(Xtr, no_of_tasks, mu = 10):
	K = []
	T = no_of_tasks
	mu_delta = np.ones((T,T))/mu
	mu_delta = mu_delta + np.diag(np.ones(T))

	for s in xrange(T):
	    kernel_t = []
	    for t in xrange(T):
	        kernel = mu_delta[s,t]*Xtr[s].dot(Xtr[t].T)
	        kernel_t.append(kernel)
	    K.append(np.hstack(kernel_t))

	return np.vstack(K)

def getSaliency(image_set, image_number):
	""" Assuming same order of images in MATLAB """
	saliency = SALIENCY[image_set][image_number]['image_saliency'][0][0]
	return saliency

def getInitialLabels(image_set, image_number, superpixels):
	saliency = getSaliency(image_set, image_number)
	# print saliency.shape
	avg_saliency = np.mean(saliency)
	# print avg_saliency

	labels = []
	new_image = np.zeros(superpixels.shape, dtype = np.uint8)
	for (i, seg) in enumerate(np.unique(superpixels)):
		# print i
		mask = np.zeros(superpixels.shape, dtype = np.uint8)
		
		mask[superpixels == seg] = 1

		no_of_pixels = np.sum(mask == 1)
		masked_saliency = saliency*mask

		less_than_avg_saliency = masked_saliency<avg_saliency
		masked_less_than_avg_saliency = less_than_avg_saliency*mask
		no_of_pixels_less_than_avg = np.sum(masked_less_than_avg_saliency)

		# print no_of_pixels, no_of_pixels_less_than_avg

		if no_of_pixels_less_than_avg > no_of_pixels/2:
			labels.append(0)
		else:
			labels.append(1)
			idx = (mask == 1)
			new_image[idx] = 1
	# plt.imshow(new_image, cmap = 'gray'); plt.show()
	return np.array(labels)

def computeTrainingData(image_set, images):
	'''image_set: numeric'''

	Xtr = {}
	Ytr = []
	superpixels = {}
	T = len(images)
	for image in xrange(T):
		feat = getFeatures(images[image])
		Xtr[image] = feat['features']
		superpixels[image] = feat['superpixels']
		lbls = getInitialLabels(image_set, image, feat['superpixels'])
		# print lbls
		Ytr.append(np.array(lbls))
	Ytr = np.concatenate(Ytr)
	return {'T':T, 'Xtr':Xtr, 'Ytr':Ytr, 'superpixels':superpixels}

def trainAndPredict(Xtr, Ytr, no_of_tasks, mu = 10, C = 1000, tol = 0.001):
	kernel = computeKernel(Xtr, no_of_tasks, mu)
	sv_machine = svm.SVC(C = C, kernel = 'precomputed', tol = tol)
	sv_machine.fit(kernel, Ytr)

	predicted_labels = sv_machine.predict(kernel)
	print np.sum(predicted_labels == 1)
	return predicted_labels

def getMasksForGrabcut(predicted_labels, no_of_tasks, superpixels):
	offset = 0
	mask = {}
	for i in xrange(no_of_tasks):
		s = superpixels[i]
		nsp = np.max(s)
		mask_i = np.zeros(s.shape, np.uint8)
		y_pred = predicted_labels[offset:offset+nsp]
		for j in xrange(nsp):
			if y_pred[j] == 1:
				mask_i[s == j+1] = 1
		mask[i] = mask_i
		offset += nsp
	return mask

def grabcut(image, mask, niters = 5):
	h, w = mask.shape
	grabcut_mask = cv2.GC_BGD*np.ones((h,w), dtype='uint8')
	for i in xrange(h):
		for j in xrange(w):
			if mask[i][j] == 1:
				grabcut_mask[i][j] = cv2.GC_FGD

	rect = cv2.boundingRect(grabcut_mask)
	if rect == (0,0,0,0):
		rect = (0,0,w,h)
	rect = (rect[0]+1, rect[1]+1, rect[2]-2, rect[3]-2)

	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)

	cv2.grabCut(image,grabcut_mask,rect,bgdModel,fgdModel,niters,cv2.GC_INIT_WITH_RECT)
	new_mask = np.where((grabcut_mask==2)|(grabcut_mask==0),0,1).astype('uint8')
	return new_mask

def computeAccuracy(grabcut_mask, ground_truth):
	ground_truth = ground_truth[:,:,0]
	diff = np.abs(grabcut_mask - ground_truth)
	size = ground_truth.size
	accuracy = (size - np.sum(diff == 1))*100.0/size
	return accuracy



##### ACTUAL COMPUTATION BEGINS HERE #####

image_sets = getImageSets(IMAGES_DIR)


N = len(image_sets)
overall_accuracy = []
for image_set in xrange(N):
	# print image_set
	print image_sets[image_set]['Name']
	images = readImages(IMAGES_DIR, image_sets[image_set], False)
	ground_truth_images = readImages(GROUND_TRUTH_DIR, image_sets[image_set], True)
	print "Images read"

	print "Computing Training Data"
	training_data = computeTrainingData(image_set, images)
	T = training_data['T']
	Xtr = training_data['Xtr']
	Ytr = training_data['Ytr']
	superpixels = training_data['superpixels']

	print "Train and Predict"
	predicted_labels = trainAndPredict(Xtr, Ytr, T)

	print "Getting Masks"
	masks = getMasksForGrabcut(predicted_labels, T, superpixels)

	accuracy = []
	for image in xrange(T):
		# print image
		grabcut_mask = grabcut(images[image], masks[image])
		acc = computeAccuracy(grabcut_mask, ground_truth_images[image])
		# print acc
		accuracy.append(acc)
	avg_accuracy = np.mean(accuracy)
	print image_set, avg_accuracy
	overall_accuracy.append(avg_accuracy)

print np.mean(overall_accuracy)
