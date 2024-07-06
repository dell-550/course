import cv2
import numpy as np
from matplotlib import pyplot as plt

from filterings import average, laplace, median, sobel, roberts
from check import get_mse_psnr_ssim


def show_2(image1, image2, title):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 2, 1)
    plt.title('opencv_deal')
    plt.imshow(image1, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title(f'{title}_image')
    plt.imshow(image2, cmap='gray')
    plt.axis('off')
    plt.savefig(f"./static/{title}.png")
    plt.show()


def show_6(image1, image2, image3, image4, image5, image6, title):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
    image4 = cv2.cvtColor(image4, cv2.COLOR_BGR2RGB)
    image5 = cv2.cvtColor(image5, cv2.COLOR_BGR2RGB)
    image6 = cv2.cvtColor(image6, cv2.COLOR_BGR2RGB)
    plt.subplot(2, 3, 1)
    plt.title('opencv_deal')
    plt.imshow(image1, cmap='gray')
    plt.axis('off')
    plt.subplot(2, 3, 4)
    plt.title(f'{title}_image')
    plt.imshow(image4, cmap='gray')
    plt.axis('off')
    plt.subplot(2, 3, 2)
    plt.title('opencv_deal')
    plt.imshow(image2, cmap='gray')
    plt.axis('off')
    plt.subplot(2, 3, 5)
    plt.title(f'{title}_image')
    plt.imshow(image5, cmap='gray')
    plt.axis('off')
    plt.subplot(2, 3, 3)
    plt.title('opencv_deal')
    plt.imshow(image3, cmap='gray')
    plt.axis('off')
    plt.subplot(2, 3, 6)
    plt.title(f'{title}_image')
    plt.imshow(image6, cmap='gray')
    plt.axis('off')
    plt.savefig(f"./static/{title}.png")
    plt.show()


def show_4(image1, image2, image3, image4, title):
    # mse, psnr, ssim = get_mse_psnr_ssim(image1, image2)
    # print("gaussian", mse, psnr, ssim)
    # mse, psnr, ssim = get_mse_psnr_ssim(image1, image3)
    # print("poisson", mse, psnr, ssim)
    # mse, psnr, ssim = get_mse_psnr_ssim(image1, image4)
    # print("salt", mse, psnr, ssim)
    cv2.imwrite(f"./static/source_{title}.png", image1)
    cv2.imwrite(f"./static/gaussian_{title}.png", image2)
    cv2.imwrite(f"./static/poisson_{title}.png", image3)
    cv2.imwrite(f"./static/salt_{title}.png", image4)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
    image4 = cv2.cvtColor(image4, cv2.COLOR_BGR2RGB)
    # image5 = cv2.cvtColor(image5, cv2.COLOR_BGR2RGB)
    # image6 = cv2.cvtColor(image6, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 4, 1)
    plt.title('source')
    plt.imshow(image1, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 4, 2)
    plt.title(f'gaussian')
    plt.imshow(image2, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.title('poisson')
    plt.imshow(image3, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.title(f'salt')
    plt.imshow(image4, cmap='gray')
    plt.axis('off')

    plt.savefig(f"./static/{title}.png")
    plt.show()
    # image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    # image3 = cv2.cvtColor(image3, cv2.COLOR_RGB2GRAY)
    # image4 = cv2.cvtColor(image4, cv2.COLOR_RGB2GRAY)
    # original_resized = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
    # mse, psnr, ssim = calculate_metrics(original_resized,image2)
    # print("gaussian", mse, psnr, ssim)
    # original_resized = cv2.resize(image1, (image3.shape[1], image3.shape[0]))
    # mse, psnr, ssim = calculate_metrics(original_resized,image2)
    # print("gaussian", mse, psnr, ssim)
    # original_resized = cv2.resize(image1, (image4.shape[1], image4.shape[0]))
    # mse, psnr, ssim = calculate_metrics(original_resized,image2)
    # print("gaussian", mse, psnr, ssim)

    # mse, psnr, ssim = calculate_metrics(image1, image3)
    # print("poisson",mse, psnr, ssim)
    #
    # mse, psnr, ssim = calculate_metrics(image1, image4)
    # print("salt",mse, psnr, ssim)


def compare_average(image1, image2, image3):
    opencv_1 = average.average(image1)
    self_1 = average.average_self(image1)
    opencv_2 = average.average(image2)
    self_2 = average.average_self(image2)
    opencv_3 = average.average(image3)
    self_3 = average.average_self(image3)
    # show_6(opencv_1, opencv_2, opencv_3, self_1, self_2, self_3, "average")
    show_2(opencv_1, self_1, "average")


def compare_median(image1, image2, image3):
    # opencv_1 = median.median(image1)
    # self_1 = median.median_self(image1)
    # show_2(opencv_1, self_1, "median")
    # opencv_2 = median.median(image2)
    # self_2 = median.median_self(image2)
    # show_2(opencv_2, self_2, "median")
    opencv_3 = median.median(image3)
    self_3 = median.median_self(image3)
    show_2(opencv_3, self_3, "median")
    # show_6(opencv_1, opencv_2, opencv_3, self_1, self_2, self_3, "median")


def compare_laplace(image1, image2, image3):
    # opencv_1 = laplace.laplace(image1)
    # self_1 = laplace.laplace_self(image1)
    # show_2(opencv_1, self_1, "laplace")
    opencv_2 = laplace.laplace(image2)
    self_2 = laplace.laplace_self(image2)
    show_2(opencv_2, self_2, "laplace")
    # opencv_3 = laplace.laplace(image3)
    # self_3 = laplace.laplace_self(image3)
    # show_2(opencv_3, self_3, "laplace")
    # show_6(opencv_1, opencv_2, opencv_3, self_1, self_2, self_3, "laplace")


def compare_sobel(image1, image2, image3):
    opencv_1 = sobel.sobel(image1)
    self_1 = sobel.sobel_self(image1)
    show_2(opencv_1, self_1, "sobel")
    # opencv_2 = sobel.sobel(image2)
    # self_2 = sobel.sobel_self(image2)
    # show_2(opencv_2, self_2, "sobel")
    # opencv_3 = sobel.sobel(image3)
    # self_3 = sobel.sobel_self(image3)
    # show_2(opencv_3, self_3, "sobel")
    # show_6(opencv_1, opencv_2, opencv_3, self_1, self_2, self_3, "sobel")


def compare_robert(image1, image2, image3):
    opencv_1 = roberts.roberts(image1)
    self_1 = roberts.roberts_self(image1)
    show_2(opencv_1, self_1, "roberts")
    # opencv_2 = roberts.roberts(image2)
    # self_2 = roberts.roberts_self(image2)
    # show_2(opencv_2, self_2, "roberts")
    # opencv_3 = roberts.roberts(image3)
    # self_3 = roberts.roberts_self(image3)
    # show_2(opencv_3, self_3, "roberts")
    # show_6(opencv_1, opencv_2, opencv_3, self_1, self_2, self_3, "roberts")


def average_deal(image1, image2, image3, image4, title):

    source = average.average_self(image1)
    gaussian_image = average.average_self(image2)
    poission_image = average.average_self(image3)
    salt_image = average.average_self(image4)

    show_4(source, gaussian_image, poission_image, salt_image, title)


def median_deal(image1, image2, image3, image4, title):
    source = median.median_self(image1)
    gaussian_image = median.median_self(image2)
    poission_image = median.median_self(image3)
    salt_image = median.median_self(image4)
    show_4(source, gaussian_image, poission_image, salt_image, title)


def sobel_deal(image1, image2, image3, image4, title):
    source = sobel.sobel_self(image1)
    gaussian_image = sobel.sobel_self(image2)
    poission_image = sobel.sobel_self(image3)
    salt_image = sobel.sobel_self(image4)
    show_4(source, gaussian_image, poission_image, salt_image, title)


def laplace_deal(image1, image2, image3, image4, title):
    source = laplace.laplace_self(image1)
    gaussian_image = laplace.laplace_self(image2)
    poission_image = laplace.laplace_self(image3)
    salt_image = laplace.laplace_self(image4)
    show_4(source, gaussian_image, poission_image, salt_image, title)


def roberts_deal(image1, image2, image3, image4, title):
    source = roberts.roberts(image1)
    gaussian_image = roberts.roberts(image2)
    poission_image = roberts.roberts(image3)
    salt_image = roberts.roberts(image4)
    show_4(source, gaussian_image, poission_image, salt_image, title)


if __name__ == "__main__":
    image_1 = cv2.imread("./datas/cat.jpg")
    image_2 = cv2.imread("./datas/flower.jpg")
    image_3 = cv2.imread("./datas/sun.jpg")

    # compare_average(image_1, image_2, image_3)
    # compare_median(image_1, image_2, image_3)
    # compare_sobel(image_1, image_2, image_3)
    # compare_laplace(image_1, image_2, image_3)
    # compare_robert(image_1, image_2, image_3)

    gaussian_image1 = cv2.imread("./static/noisy/gaussian_image1.png")
    gaussian_image2 = cv2.imread("./static/noisy/gaussian_image2.png")
    gaussian_image3 = cv2.imread("./static/noisy/gaussian_image3.png")

    poission_image1 = cv2.imread("./static/noisy/poission_image1.png")
    poission_image2 = cv2.imread("./static/noisy/poission_image2.png")
    poission_image3 = cv2.imread("./static/noisy/poission_image3.png")

    salt_image1 = cv2.imread("./static/noisy/salt_image1.png")
    salt_image2 = cv2.imread("./static/noisy/salt_image2.png")
    salt_image3 = cv2.imread("./static/noisy/salt_image3.png")

    # average_deal(image_1, gaussian_image1, poission_image1, salt_image1, "cat")
    # average_deal(image_2, gaussian_image2, poission_image2, salt_image2, "flower")
    # average_deal(image_3, gaussian_image3, poission_image3, salt_image3, "sun")
    # #
    # roberts_deal(image_1, gaussian_image1, poission_image1, salt_image1, "cat")
    # roberts_deal(image_2, gaussian_image2, poission_image2, salt_image2, "flower")
    # roberts_deal(image_3, gaussian_image3, poission_image3, salt_image3, "sun")
    # #
    # laplace_deal(image_1, gaussian_image1, poission_image1, salt_image1, "cat")
    # laplace_deal(image_2, gaussian_image2, poission_image2, salt_image2, "flower")
    # laplace_deal(image_3, gaussian_image3, poission_image3, salt_image3, "sun")
    # # #
    # sobel_deal(image_1, gaussian_image1, poission_image1, salt_image1, "cat")
    # sobel_deal(image_2, gaussian_image2, poission_image2, salt_image2, "flower")
    # sobel_deal(image_3, gaussian_image3, poission_image3, salt_image3, "sun")
    #
    median_deal(image_1, gaussian_image1, poission_image1, salt_image1, "cat")
    median_deal(image_2, gaussian_image2, poission_image2, salt_image2, "flower")
    median_deal(image_3, gaussian_image3, poission_image3, salt_image3, "sun")
