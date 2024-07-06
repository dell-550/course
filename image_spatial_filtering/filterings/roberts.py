import cv2
import numpy as np


def roberts_cross_channel(channel):
    Kx = np.array([[1, 0], [0, -1]], dtype=int)
    Ky = np.array([[0, 1], [-1, 0]], dtype=int)
    # Convolution
    Gx = cv2.filter2D(channel, -1, Kx)
    Gy = cv2.filter2D(channel, -1, Ky)
    G = np.sqrt(Gx ** 2 + Gy ** 2)
    G = G / G.max() * 255  # Normalize to 0-255
    return G.astype(np.uint8)


def roberts(img):
    channels = cv2.split(img)
    edges = [roberts_cross_channel(channel) for channel in channels]
    edges = cv2.merge(edges)
    image_float = img.astype(float)
    edges_float = edges.astype(float)
    sharpened = image_float + edges_float
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened

def custom_roberts_filter(image):
    roberts_kernel_x = np.array([[1, 0],     # 定义Robert核
                                 [0, -1]])
    roberts_kernel_y = np.array([[0, 1],
                                 [-1, 0]])
    image_height, image_width = image.shape
    gradient_magnitude = np.zeros_like(image, dtype=np.float64)
    for i in range(image_height - 1):      # Robert滤波器卷积操作
        for j in range(image_width - 1):
            gx = np.sum(roberts_kernel_x * image[i:i + 2, j:j + 2])
            gy = np.sum(roberts_kernel_y * image[i:i + 2, j:j + 2])
            gradient_magnitude[i, j] = np.sqrt(gx ** 2 + gy ** 2)
    return gradient_magnitude

def roberts_self(image):
    blue_channel, green_channel, red_channel = cv2.split(image)

    # 对每个通道应用自定义的Robert滤波器
    filtered_blue_channel = custom_roberts_filter(blue_channel)
    filtered_green_channel = custom_roberts_filter(green_channel)
    filtered_red_channel = custom_roberts_filter(red_channel)

    # 合并通道
    filtered_image = cv2.merge([filtered_blue_channel, filtered_green_channel, filtered_red_channel])

    # 将结果归一化到0-255并转换为uint8类型
    filtered_image = np.uint8(np.clip(filtered_image, 0, 255))
    filtered_image = image + filtered_image
    sharpened = np.clip(filtered_image, 0, 255).astype(np.uint8)
    return sharpened



if __name__ == "__main__":
    image = cv2.imread("E:\python\image_spatial_filtering\datas\cat.jpg")
    shenr = roberts_self(image)
    cv2.imwrite("E:\python\image_spatial_filtering\datas\sobel.jpg", shenr)
