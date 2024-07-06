import cv2
import numpy as np


def laplace(image):
    # 将图像转换为灰度图像（如果不是灰度图像）
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 应用 Laplace 算子
    laplacian = cv2.Laplacian(gray_image, cv2.CV_16S, ksize=3)
    # 将结果转换回 8 位图像
    sharp_image = cv2.convertScaleAbs(laplacian)
    # 将锐化后的灰度图像叠加回原始彩色图像
    sharpened_image = cv2.addWeighted(image, 1, cv2.cvtColor(sharp_image, cv2.COLOR_GRAY2BGR), 1, 0)
    return sharpened_image


def apply_kernel(image, kernel):
    # 获取图像的尺寸
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    # 创建一个零填充后的图像，以便处理边界
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    # 创建输出图像
    output_image = np.zeros_like(image, dtype=np.float32)
    # 执行卷积操作
    for i in range(image_height):
        for j in range(image_width):
            window = padded_image[i:i + kernel_height, j:j + kernel_width]
            output_image[i, j] = np.sum(window * kernel)
    return output_image


def laplace_self(image):
    # 将图像转换为灰度图像（如果不是灰度图像）
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 定义 Laplace 核
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=int)
    # 手动应用 Laplace 核
    laplacian = apply_kernel(gray_image, kernel)
    # 将结果转换回 8 位图像
    sharp_image = cv2.convertScaleAbs(laplacian)
    # 将锐化后的灰度图像叠加回原始彩色图像
    sharpened_image = cv2.addWeighted(image, 1, cv2.cvtColor(sharp_image, cv2.COLOR_GRAY2BGR), 1, 0)
    return sharpened_image
