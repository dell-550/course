# encoding:utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt

from noisy import add_poisson_noise, add_salt_and_pepper_noise, add_gaussian_noise
from filtering import average_filter, median_filter, sharpen_filter, laplacian_filter


def average(image):
    return cv2.cvtColor(cv2.blur(image, (5, 5)), cv2.COLOR_BGR2RGB)


def median(image):
    return cv2.cvtColor(cv2.medianBlur(image, 5), cv2.COLOR_BGR2RGB)


def sharpen(image):
    # 创建锐化滤波器的卷积核
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    # 应用锐化滤波器
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image


def laplacian(image):
    # 应用拉普拉斯算子
    laplacian_img = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
    laplacian_img = cv2.convertScaleAbs(laplacian_img)
    # 锐化图像
    sharpened_img = cv2.addWeighted(image, 1.5, laplacian_img, -0.5, 0)
    return sharpened_img


def show(source, gaussian, salt, poisson, filter_name):
    data = "data_1"
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 4, 1)
    plt.title('source_' + filter_name)
    plt.imshow(source, cmap='gray')
    cv2.imwrite(f'./static/{data}/source_{filter_name}.png', gaussian)
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title('gaussian_' + filter_name)
    plt.imshow(gaussian, cmap='gray')
    cv2.imwrite(f'./static/{data}/gaussian_{filter_name}.jpg', gaussian)
    plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.title('salt_' + filter_name)
    cv2.imwrite(f'./static/{data}/salt_{filter_name}.jpg', salt)
    plt.imshow(salt, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.title('poisson_' + filter_name)
    cv2.imwrite(f'./static/{data}/poisson_{filter_name}.jpg', poisson)
    plt.imshow(poisson, cmap='gray')
    plt.axis('off')
    plt.savefig(f"./static/{filter_name}.png")
    # plt.show()


# 读取图像
source_image = cv2.imread(r'E:\data\R.jpg')

# gaussian_image = add_gaussian_noise(source_image)
# salt_image = add_salt_and_pepper_noise(source_image)
# poisson_image = add_poisson_noise(source_image)
# show(source_image, gaussian_image, salt_image, poisson_image, filter_name="source")
# cv2.imwrite('./static/gaussian_image.png', gaussian_image)
# cv2.imwrite('./static/salt_image.png', salt_image)
# cv2.imwrite('./static/poisson_image.png', poisson_image)
image1 = average(source_image)
image2 = median(source_image)
cv2.imwrite('./static/image1.png', image1)
cv2.imwrite('./static/image2.png', image2)
# # 应用均值滤波器
# source_average = average(source_image)
# gaussian_average = average(gaussian_image)
# salt_average = average(salt_image)
# poisson_average = average(poisson_image)
# show(source_average, gaussian_average, salt_average, poisson_average, filter_name="average")
# #
# source_self_average = average_filter(source_image)
# gaussian_self_average = average_filter(gaussian_image)
# salt_self_average = average_filter(salt_image)
# poisson_self_average = average_filter(poisson_image)
# show(source_self_average, gaussian_self_average, salt_self_average, poisson_self_average, filter_name="average_self")
#
# # media
# source_media = median(source_image)
# gaussian_media = median(gaussian_image)
# salt_media = median(salt_image)
# poisson_media = median(poisson_image)
# show(source_media, gaussian_media, salt_media, poisson_media, filter_name="median")
# #
# source_self_media = median_filter(source_image)
# gaussian_self_media = median_filter(gaussian_image)
# salt_self_media = median_filter(salt_image)
# poisson_self_media = median_filter(poisson_image)
# show(source_self_media, gaussian_self_media, salt_self_media, poisson_self_media, filter_name="median_self")
# # sharpen
# source_sharpen = sharpen(source_image)
# gaussian_sharpen = sharpen(gaussian_image)
# salt_sharpen = sharpen(salt_image)
# poisson_sharpen = sharpen(poisson_image)
# show(source_sharpen, gaussian_sharpen, salt_sharpen, poisson_sharpen, filter_name="sharpen")

# source_self_sharpen = sharpen_filter(source_image)
# gaussian_self_sharpen = sharpen_filter(gaussian_image)
# salt_self_sharpen = sharpen_filter(salt_image)
# poisson_self_sharpen = sharpen_filter(poisson_image)
# show(source_self_sharpen, gaussian_self_sharpen, salt_self_sharpen, poisson_self_sharpen, filter_name="sharpen_self")
# laplacian
# source_laplacian = laplacian(source_image)
# gaussian_laplacian = laplacian(gaussian_image)
# salt_laplacian = laplacian(salt_image)
# poisson_laplacian = laplacian(poisson_image)
# show(source_laplacian, gaussian_laplacian, salt_laplacian, poisson_laplacian, filter_name="laplacian")

# source_self_laplacian = laplacian_filter(source_image)
# gaussian_self_laplacian = laplacian_filter(gaussian_image)
# salt_self_laplacian = laplacian_filter(salt_image)
# poisson_self_laplacian = laplacian_filter(poisson_image)
# show(source_self_laplacian, gaussian_self_laplacian, salt_self_laplacian, poisson_self_laplacian, filter_name="laplacian_self")

# 显示原始图像和均值滤波后的图像

#
# show(source_average, gaussian_average, salt_average, poisson_average, filter_name="average")
# show(source_media, gaussian_media, salt_media, poisson_media, filter_name="median")
# show(source_sharpen, gaussian_sharpen, salt_sharpen, poisson_sharpen, filter_name="sharpen")
# show(source_laplacian, gaussian_laplacian, salt_laplacian, poisson_laplacian, filter_name="laplacian")
#
# show(source_self_average, gaussian_self_average, salt_self_average, poisson_self_average, filter_name="self_average")
# show(source_self_media, gaussian_self_media, salt_self_media, poisson_self_media, filter_name="self_median")
# show(source_self_sharpen, gaussian_self_sharpen, salt_self_sharpen, poisson_self_sharpen, filter_name="self_sharpen")
# show(source_self_laplacian, gaussian_self_laplacian, salt_self_laplacian, poisson_self_laplacian, filter_name="self_laplacian")
