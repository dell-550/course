import cv2
import numpy as np


def average(image):
    blur_image = cv2.blur(image, (5, 5))
    return blur_image


def average_self(image, kernel_size=3):
    # 创建均值滤波器的卷积核
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
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
                filtered_image[i, j, channel] = np.sum(
                    padded_image[i:i + kernel_size, j:j + kernel_size, channel] * kernel)
    return filtered_image
