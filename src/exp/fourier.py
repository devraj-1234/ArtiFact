import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load image in grayscale
img = cv2.imread("unnamed.png", 0)   # replace with your image path

# 2. Apply 2D FFT
f = np.fft.fft2(img)

# 3. Shift the zero-frequency component to the center
fshift = np.fft.fftshift(f)

# 4. Compute magnitude spectrum (log scale for visibility)
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

# 5. Plot results
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(magnitude_spectrum, cmap="gray")
plt.title("FFT Magnitude Spectrum")
plt.axis("off")

plt.show()
