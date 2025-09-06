import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load grayscale image
img = cv2.imread("your_image.jpg", 0)

# 2. Apply 2D FFT
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)  # shift zero freq to center

# 3. Compute magnitude spectrum
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

# 4. Display original and spectrum
plt.subplot(121), plt.imshow(img, cmap="gray")
plt.title("Original Image"), plt.axis("off")

plt.subplot(122), plt.imshow(magnitude_spectrum, cmap="gray")
plt.title("FFT Magnitude Spectrum"), plt.axis("off")

plt.show()
