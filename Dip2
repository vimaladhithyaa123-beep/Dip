import cv2
import numpy as np

high_res_factor = 1.0
low_res_factor = 0.25

img = cv2.imread("watch.jpg", cv2.IMREAD_GRAYSCALE)
original = img.copy()

sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
gradient_magnitude = np.sqrt(sobel_x*2 + sobel_y*2)
gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

_, detail_mask = cv2.threshold(gradient_magnitude, 40, 255, cv2.THRESH_BINARY)

low_res = cv2.resize(img, None, fx=low_res_factor, fy=low_res_factor, interpolation=cv2.INTER_AREA)
low_res_upsampled = cv2.resize(low_res, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

mask = detail_mask > 0
adaptive_quantized = low_res_upsampled.copy()
adaptive_quantized[mask] = img[mask]

cv2.imshow("Original", original)
cv2.imshow("Gradient Map", gradient_magnitude)
cv2.imshow("Low-Resolution Interpolated", low_res_upsampled)
cv2.imshow("Adaptive Non-Uniform Quantized Image", adaptive_quantized)

cv2.imwrite("adaptive_non_uniform_output.jpg", adaptive_quantized)

cv2.waitKey(0)
cv2.destroyAllWindows()
