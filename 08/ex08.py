import numpy as np
import imageio
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from skimage.color import rgb2gray

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter


# Load picture and detect edges
image = imageio.imread('coins.jpg')
image = rgb2gray(image)
edges = canny(image, sigma=1)

plt.imshow(edges, cmap='gray')
plt.show()

# Detect coins
im_resolution = 0.12 # mm/pixel
coin_diameter_mm = 24.5 # mm, 5 markka
# coin_diameter_mm = 27.25 # mm, 10 markka (uncomment to use)

coin_radius_pixels = (coin_diameter_mm / 2) / im_resolution
print("Radius of the coin in pixels is {}".format(coin_radius_pixels))
hough_res = hough_circle(edges, radius=coin_radius_pixels) # note shape is (1, image.shape)

plt.imshow(hough_res[0], cmap='gray')
plt.show()

# Select the most prominent circles, draw
accums, cx, cy, radii = hough_circle_peaks(hough_res, [coin_radius_pixels], total_num_peaks=2)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
for xx, yy, radius in zip(cx, cy, radii):
    circ = Circle((xx,yy), radius, color='red', alpha=0.2) # use alpha for transparency
    ax.add_patch(circ)

ax.imshow(image, cmap=plt.cm.gray)
plt.show()