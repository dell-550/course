import cv2
import numpy as np


def median(image):
    blur_image = cv2.medianBlur(image, 5)
    return blur_image


def median_self(image, kernel_size=3):
    # 获取图像尺寸
    h, w, c = image.shape
    # 为图像添加边界填充
    padded_image = np.pad(image, ((kernel_size // 2, kernel_size // 2),
                                  (kernel_size // 2, kernel_size // 2),
                                  (0, 0)), 'constant')
    filtered_image = np.zeros_like(image)
    for channel in range(c):
        for i in range(h):
            for j in range(w):
                filtered_image[i, j, channel] = np.median(padded_image[i:i + kernel_size, j:j + kernel_size, channel])
    return filtered_image
