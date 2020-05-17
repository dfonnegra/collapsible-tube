import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

filename = '../img/disks-1-noise.jpg'

rows, cols = (768, 768)
img = cv2.imread(filename)
img = cv2.resize(img, dsize=(rows, cols))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Finds the center of the disk
img = cv2.GaussianBlur(img, (9, 9), sigmaX=2, sigmaY=2)
circles = cv2.HoughCircles(
    img,
    cv2.HOUGH_GRADIENT,
    1,
    minDist=100,
    minRadius=rows // 20
)
circle = circles[0, np.argmax(circles[0, :, 2]), :]


cent = (circle[0], circle[1])
max_radius = circle[2]
# plt.gcf().gca().add_artist(plt.Circle(cent, max_radius, color='r', fill=False))
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()
polar = cv2.linearPolar(img, cent, max_radius, cv2.WARP_FILL_OUTLIERS)
dft = cv2.dft(np.float32(polar), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude = np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))
mask = np.zeros((*magnitude.shape, 2))
mask[(magnitude < 0.6) & (magnitude > 0.5)] = 1
dft_shift = dft_shift * mask
f_ishift = np.fft.ifftshift(dft_shift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

#plt.imshow(20*np.log(cv2.magnitude(dft[:,:,0], dft[:,:,1])))
#plt.imshow(20*np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(img_back, cmap='gray')
#plt.imshow(polar)
plt.show()
#cv2.waitKey(0)