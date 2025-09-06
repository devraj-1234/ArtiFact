import cv2
import matplotlib.pyplot as plt

img = cv2.imread("sample_img.png")  # 0 = grayscale
'''
plt.imshow(img, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()'''

print("Image shape:", img.shape)
print("Pixel value at (50,50):", img[50,50])

# Blur (low-pass filter in spatial domain)
blur = cv2.GaussianBlur(img, (5,5), 0)

# Sharpen (using kernel)
import numpy as np
kernel = np.array([[0,0,0],
                   [0,0,0],
                   [0,0,0]])
sharpen = cv2.filter2D(img, -1, kernel)

plt.subplot(131), plt.imshow(img, cmap="gray"), plt.title("Original")
plt.subplot(132), plt.imshow(blur, cmap="gray"), plt.title("Blur")
plt.subplot(133), plt.imshow(sharpen, cmap="gray"), plt.title("Sharpen")
plt.show()
