import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


def resize_images_to_fixed_size(imageA, imageB, size=(512, 512)):
    # 调整图像大小到固定尺寸
    imageA_resized = cv2.resize(imageA, size, interpolation=cv2.INTER_AREA)
    imageB_resized = cv2.resize(imageB, size, interpolation=cv2.INTER_AREA)
    return imageA_resized, imageB_resized


def calculate_mse(imageA, imageB):
    # 确保图像大小一致
    assert imageA.shape == imageB.shape, "Images must have the same dimensions"

    # 计算均方误差
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err


def calculate_psnr(imageA, imageB):
    # 计算均方误差
    mse_value = calculate_mse(imageA, imageB)

    # 如果MSE为零，则表示两张图像完全相同
    if mse_value == 0:
        return float('inf')

    # 计算PSNR
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse_value))

    return psnr_value


def calculate_ssim(imageA, imageB):
    # 确保图像大小一致并且是灰度图像
    assert imageA.shape == imageB.shape, "Images must have the same dimensions"

    # 计算SSIM
    ssim_value = ssim(imageA, imageB, data_range=imageB.max() - imageB.min())

    return ssim_value


def get_mse_psnr_ssim(imageA, imageB):
    imageA, imageB = resize_images_to_fixed_size(imageA, imageB, size=(512, 512))
    mse_value = calculate_mse(imageA, imageB)
    psnr_value = calculate_psnr(imageA, imageB)
    ssim_value = calculate_ssim(imageA, imageB)
    return mse_value, psnr_value, ssim_value
# 示例使用方法
if __name__ == "__main__":
    # 读取图像并转换为灰度图像
    imageA = cv2.imread(r'E:\python\image_spatial_filtering\static\sobel\source_flower.png', cv2.IMREAD_GRAYSCALE)
    imageB = cv2.imread(r'E:\python\image_spatial_filtering\static\sobel\poisson_flower.png', cv2.IMREAD_GRAYSCALE)

    # 调整图像大小到固定尺寸
    mse_value, psnr_value, ssim_value = get_mse_psnr_ssim(imageA, imageB)
    print(f"MSE: {mse_value}")
    print(f"PSNR: {psnr_value}")
    print(f"SSIM: {ssim_value}")