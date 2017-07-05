from skimage.io import imread, imshow, imsave
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from numpy.linalg import norm
from scipy.spatial.distance import pdist, cdist, squareform
from numpy.linalg import solve

img = imread('baboon.png', as_grey = 1)
clrimg = imread('baboon.png')

d = np.zeros(img.shape)
d[70:72, 75:250] = 1
d[200:202, 110:390] = 1
d[450:452, 310:410] = 1

def getNeighbourhood(intensity_image, r):
	nbrhood = OrderedDict()
	h, w = intensity_image.shape
	padded_image = np.zeros((h+2*r, w+2*r))
	padded_image[r:r+h, r:r+w] = intensity_image
	for i in xrange(r,r+h):
		for j in xrange(r,r+w):
			nbrhood[(i-r,j-r)] = padded_image[i-r:i+r+1, j-r:j+r+1].ravel()
	return nbrhood

def getDomain(clrimg, domain_mask):
	h,w = domain_mask.shape
	domain = OrderedDict()
	for i in xrange(h):
		for j in xrange(w):
			if domain_mask[i][j] == 1:
				domain[(i,j)] = clrimg[i,j,:]
	return domain

r1 = 0

nbrhood = getNeighbourhood(img, r1)
domain_color = getDomain(clrimg, d)

domain_intensity = OrderedDict()
for i in domain_color:
	domain_intensity[i] = nbrhood[i]

r = np.array([domain_color[i][0] for i in domain_color])
g = np.array([domain_color[i][1] for i in domain_color])
b = np.array([domain_color[i][2] for i in domain_color])

print "rgb"

m = len(domain_color)
eye = np.eye(m)
power = 2
s1 = 0.001
s2 = 10
rho = norm(img.shape)

def getKd(domain, power, s1, s2, rho, r):
	d1 = squareform(pdist(domain.values(), lambda x, y: norm(x - y)**power))
	d2 = squareform(pdist(domain.keys(), lambda x, y: norm(x - y)**power))
	
	rho_p = rho**power
	r_p = 2*((2*r+1)**power)

	e1 = np.exp(-d1/(r_p*s1))
	e2 = np.exp(-d2/(rho_p*s2))
	return e1*e2

kd = getKd(domain_intensity, power, s1, s2, rho, r1)
print "kd"

# Adaptive regularisation parameter selection

a = 0.00001
q = 1.5
gamma_list = [a*(q**i) for i in xrange(20)]
F = []
for gamma in gamma_list:
	s = kd+gamma*m*eye
	A1 = solve(s, r)
	A2 = solve(s, g)
	A3 = solve(s, b)
	F1 = np.hstack((A1, A2, A3))
	F.append(F1)

distance_norm = [norm(F[i] - F[i-1]) for i in xrange(1, 20)]
min_index = distance_norm.index(min(distance_norm))
gamma = gamma_list[min_index]

print "solved"
print "Gamma: " + str(gamma)

counter = 0
red = np.zeros(img.size)
green = np.zeros(img.size)
blue = np.zeros(img.size)

for i in nbrhood:
	if counter%1000 == 0:
		print counter
	d1 = cdist(domain_intensity.values(),[nbrhood[i]], lambda x, y: norm(x-y)**power)
	d2 = cdist(domain_intensity.keys(),[i], lambda x, y: norm(x-y)**power)
	rho_p = rho**power
	r_p = 2*((2*r1+1)**power)

	e1 = np.exp(-d1/(r_p*s1))
	e2 = np.exp(-d2/(rho_p*s2))
	kcd = (e1*e2).ravel()
	red[counter] = kcd.dot(A1)
	green[counter] = kcd.dot(A2)
	blue[counter] = kcd.dot(A3)

	counter+=1

red = np.reshape(red, img.shape).astype('uint8')
green = np.reshape(green, img.shape).astype('uint8')
blue = np.reshape(blue, img.shape).astype('uint8')

new_img = np.dstack((red, green, blue))
save_path = str(gamma) + ".png"
imsave(save_path, new_img)

