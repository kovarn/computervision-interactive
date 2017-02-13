import cv2
from matplotlib import pyplot as plt

import filters

image_file = "apple.jpg"
kernel_size = 7
src = cv2.imread(image_file)
if len(src.shape) == 3:
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

kuwahara = filters.kuwahara_filter(src, kernel_size)
plt.subplot(2, 2, 1)
plt.imshow(src, cmap=plt.cm.gray)  # "Original",
plt.title("Original")
plt.subplot(2, 2, 2)
plt.imshow(kuwahara, cmap=plt.cm.gray)  # "Kuwahara",
plt.subplot(2, 2, 3)
plt.title("Kuwahara")
plt.imshow(cv2.Canny(src, 70, 150), cmap=plt.cm.gray)  # "Canny before filter",
plt.subplot(2, 2, 4)
plt.title("Canny before filter")
plt.imshow(cv2.Canny(kuwahara, 70, 150), cmap=plt.cm.gray)  # "Canny after filter",
plt.title("Canny after filter")
plt.axis('off')
plt.show()
