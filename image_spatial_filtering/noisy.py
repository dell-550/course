import cv2
import numpy as np
from matplotlib import pyplot as plt


def add_gaussian_noise(image, mean=0, var=0.01):
    """
    添加高斯噪声到图像
    :param image: 输入图像，灰度图
    :param mean: 噪声均值
    :param var: 噪声方差
    :return: 含高斯噪声的图像
    """
    # 将图像转换为浮点型
    image = image.astype(np.float32)
    # 生成高斯噪声
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape)
    # 添加噪声到图像
    noisy_image = image + gauss * 255
    # 将结果裁剪到 [0, 255] 区间，并转换为 uint8 类型
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


# 添加椒盐噪声
def add_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):
    h, w = image.shape[:2]
    noisy_image = np.copy(image)
    # 添加盐噪声
    num_salt = np.ceil(salt_prob * h * w)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255
    # 添加胡椒噪声
    num_pepper = np.ceil(pepper_prob * h * w)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0
    return noisy_image


def add_poisson_noise(image):
    """
    添加泊松噪声到图像
    :param image: 输入图像，灰度图
    :return: 含泊松噪声的图像
    """
    # 将图像转换为浮点型并放大像素值范围
    image = image.astype(np.float32)
    # 增强图像亮度，使得泊松噪声更明显
    image = image * 255.0 / image.max()
    # 生成泊松噪声
    noisy = np.random.poisson(image)*1.5
    # 将结果裁剪到 [0, 255] 区间，并转换为 uint8 类型
    noisy_image = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy_image



if __name__ == '__main__':
    # source_image = cv2.imread(r'E:\data\moon.jpg', cv2.IMREAD_GRAYSCALE)
    source_image = cv2.imread('E:/data/R.jpg', cv2.IMREAD_GRAYSCALE)
    gaussian_noisy_image = add_gaussian_noise(source_image)
    salt_pepper_noisy_image = add_salt_and_pepper_noise(source_image)
    poisson_noisy_image = add_poisson_noise(source_image)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 4, 1)
    plt.title('source')
    plt.imshow(source_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title('gauss')
    plt.imshow(gaussian_noisy_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title('salt')
    plt.imshow(salt_pepper_noisy_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title('poisson')
    plt.imshow(poisson_noisy_image, cmap='gray')
    plt.axis('off')

    plt.savefig('./static/noisy.png')
    plt.show()

    cv2.imshow('Gaussian Noise', gaussian_noisy_image)
    cv2.imshow('Salt and Pepper Noise', salt_pepper_noisy_image)
    cv2.imshow('Poisson Noise', poisson_noisy_image)
    cv2.imwrite('Gaussian.jpg', gaussian_noisy_image)
    cv2.imwrite('Salt.jpg', salt_pepper_noisy_image)
    cv2.imwrite('Poisson.jpg', poisson_noisy_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()