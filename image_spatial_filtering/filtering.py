import cv2
import numpy as np
from matplotlib import pyplot as plt




def show(image1,image2,image3,image4,image5,image6,image7,image8, filter_name):
    plt.figure(figsize=(10, 5))
    plt.axis('off')
    plt.subplot(2, 4, 1)
    plt.title('1' + filter_name)
    plt.imshow(image1, cmap='gray')
    plt.subplot(2, 4, 2)
    plt.title('2' + filter_name)
    plt.imshow(image2, cmap='gray')
    plt.subplot(2, 4, 3)
    plt.title('source_' + filter_name)
    plt.imshow(image3, cmap='gray')
    plt.subplot(2, 4, 4)
    plt.title('source_' + filter_name)
    plt.imshow(image4, cmap='gray')
    plt.subplot(2, 4, 5)
    plt.title('source_' + filter_name)
    plt.imshow(image5, cmap='gray')
    plt.subplot(2, 4, 6)
    plt.title('source_' + filter_name)
    plt.imshow(image6, cmap='gray')
    plt.subplot(2, 4, 7)
    plt.title('source_' + filter_name)
    plt.imshow(image7, cmap='gray')
    plt.subplot(2, 4, 8)
    plt.title('source_' + filter_name)
    plt.imshow(image8, cmap='gray')
    # plt.savefig(f"./static/{filter_name}.png")
    plt.show()


def average_filter(image, kernel_size=3):
    # 创建均值滤波器的卷积核
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    # 获取图像尺寸
    h, w = image.shape
    # 为图像添加边界填充
    padded_image = np.pad(image, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2)),
                          'constant')
    filtered_image = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            filtered_image[i, j] = np.sum(padded_image[i:i + kernel_size, j:j + kernel_size] * kernel)
    return filtered_image


def median_filter(image, kernel_size=3):
    # 获取图像尺寸
    h, w = image.shape
    # 为图像添加边界填充
    padded_image = np.pad(image, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2)),
                          'constant')
    # 输出图像
    filtered_image = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            filtered_image[i, j] = np.median(padded_image[i:i + kernel_size, j:j + kernel_size])

    return filtered_image


def sharpen_filter(image):
    image = average_filter(image)
    # 创建锐化滤波器的卷积核
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    # 获取图像尺寸
    h, w = image.shape
    # 为图像添加边界填充
    padded_image = np.pad(image, ((1, 1), (1, 1)), 'constant')
    # 输出图像
    filtered_image = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            filtered_image[i, j] = np.sum(padded_image[i:i + 3, j:j + 3] * kernel)
    return filtered_image


# laplacian filter

def padding(img, K_size=3):
    # img 为需要处理图像
    # K_size 为滤波器也就是卷积核的尺寸，这里我默认设为3*3，基本上都是奇数
    # 获取图片尺寸
    H, W = img.shape
    pad = K_size // 2  # 需要在图像边缘填充的0行列数，
    # 之所以我要这样设置，是为了处理图像边缘时，滤波器中心与边缘对齐
    # 先填充行
    rows = np.zeros((pad, W), dtype=np.uint8)
    # 再填充列
    cols = np.zeros((H + 2 * pad, pad), dtype=np.uint8)
    # 进行拼接
    img = np.vstack((rows, img, rows))  # 上下拼接
    img = np.hstack((cols, img, cols))  # 左右拼接
    return img


# Prewitt 滤波函数
def laplacian_filter(img, K_size=3):
    img = average_filter(img)
    # 获取图像尺寸
    H, W = img.shape
    # 进行padding
    pad = K_size // 2
    out = padding(img, K_size=3)
    # 滤波器系数
    K = np.array([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
    tem = out.copy()  # 进行滤波
    for h in range(H):
        for w in range(W):
            out[pad + h, pad + w] = (-1) * np.sum(K * tem[h:h + K_size, w:w + K_size]) + tem[
                pad + h, pad + w]
    out = np.clip(out, 0, 255)
    out = out[pad:pad + H, pad:pad + W].astype(np.uint8)
    return out


if __name__ == '__main__':
    source_image = cv2.imread(r'E:\data\R.jpg', cv2.IMREAD_GRAYSCALE)
    gaussian_image = cv2.imread('./static/gaussian_image.png', cv2.IMREAD_GRAYSCALE)
    salt_image = cv2.imread('./static/salt_image.png', cv2.IMREAD_GRAYSCALE)
    poisson_image = cv2.imread('./static/poisson_image.png', cv2.IMREAD_GRAYSCALE)

    laplacian_image1 = laplacian_filter(source_image)
    sharpened_image2 = sharpen_filter(source_image)

    laplacian_image3 = laplacian_filter(gaussian_image)
    sharpened_image4 = sharpen_filter(gaussian_image)

    laplacian_image5 = laplacian_filter(salt_image)
    sharpened_image6 = sharpen_filter(salt_image)

    laplacian_image7 = laplacian_filter(poisson_image)
    sharpened_image8 = sharpen_filter(poisson_image)

    show(laplacian_image1,sharpened_image2,laplacian_image3,sharpened_image4,laplacian_image5,sharpened_image6,laplacian_image7,sharpened_image8,filter_name="a")

    # median_filtered_image = median_filter(source_image)
    # filtered_image = average_filter(source_image)
    # cv2.imwrite('./static/sharpened_image.png', sharpened_image)
    # cv2.imwrite('./static/median_filtered_image.png', median_filtered_image)
    # cv2.imwrite('./static/average_filter.png', filtered_image)
    # cv2.imwrite('./static/laplacian_image.png', laplacian_image)
    # show(laplacian_image, sharpened_image, median_filtered_image, filtered_image, filter_name="a")
