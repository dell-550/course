import cv2
import numpy as np


def sobel_deal(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_magnitude = cv2.normalize(grad_magnitude, grad_magnitude, 0, 255, cv2.NORM_MINMAX)
    grad_magnitude = grad_magnitude.astype(np.uint8)
    sharpened_image = cv2.addWeighted(image, 1, grad_magnitude, 1, 0)
    return sharpened_image


def sobel(img):
    channels = cv2.split(img)
    edges = [sobel_deal(channel) for channel in channels]
    edges = cv2.merge(edges)
    image_float = img.astype(float)
    edges_float = edges.astype(float)
    sharpened = image_float + edges_float
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened


def sobel_self(image):
    channels = cv2.split(image)
    edges = [sobel_deals(channel) for channel in channels]
    edges = cv2.merge(edges)
    image_float = image.astype(float)
    edges_float = edges.astype(float)
    sharpened = image_float + edges_float
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened


def sobel_deals(image):
    grad_x = custom_sobel_x(image)
    grad_y = custom_sobel_y(image)
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_magnitude = cv2.normalize(grad_magnitude, grad_magnitude, 0, 255, cv2.NORM_MINMAX)
    grad_magnitude = grad_magnitude.astype(np.uint8)
    sharpened_image = cv2.addWeighted(image, 1, grad_magnitude, 1, 0)
    return sharpened_image


def custom_sobel_x(image):
    sobel_kernel_x = np.array([      # 定义Sobel核，用于水平方向的边缘检测
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1] ])
    image_height, image_width = image.shape
    assert len(image.shape) == 2, "Input image must be grayscale"
    pad_height = sobel_kernel_x.shape[0] // 2
    pad_width = sobel_kernel_x.shape[1] // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    sobel_x_image = np.zeros_like(image, dtype=np.float64)
    for i in range(image_height):     # 卷积操作
        for j in range(image_width):
            region = padded_image[i:i + sobel_kernel_x.shape[0], j:j + sobel_kernel_x.shape[1]]
            sobel_x_value = np.sum(region * sobel_kernel_x)
            sobel_x_image[i, j] = sobel_x_value
    return sobel_x_image


def custom_sobel_y(image):
    # 定义Sobel核，用于垂直方向的边缘检测
    sobel_kernel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    # 获取图像尺寸
    image_height, image_width = image.shape

    # 确保输入图像是灰度图像
    assert len(image.shape) == 2, "Input image must be grayscale"

    # 填充图像
    pad_height = sobel_kernel_y.shape[0] // 2
    pad_width = sobel_kernel_y.shape[1] // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

    # 创建输出图像
    sobel_y_image = np.zeros_like(image, dtype=np.float64)

    # 卷积操作
    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i + sobel_kernel_y.shape[0], j:j + sobel_kernel_y.shape[1]]
            sobel_y_value = np.sum(region * sobel_kernel_y)
            sobel_y_image[i, j] = sobel_y_value

    return sobel_y_image


if __name__ == "__main__":
    image = cv2.imread("E:\python\image_spatial_filtering\datas\cat.jpg")
    shenr = sobel_self(image)
    cv2.imwrite("E:\python\image_spatial_filtering\datas\sobel.jpg", shenr)
