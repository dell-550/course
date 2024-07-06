import cv2
import numpy as np
from matplotlib import pyplot as plt

from noisy import add_poisson_noise, add_gaussian_noise, add_salt_and_pepper_noise
from filterings import average, laplace, median, sobel ,roberts

def show_2(image1, image2,title):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 2, 1)
    plt.title('source')
    plt.imshow(image1, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title(f'{title}_image')
    plt.imshow(image2, cmap='gray')
    plt.axis('off')
    plt.savefig(f"./static/{title}.png")
    plt.show()

def add_noisy(image1,image2,image3):
    poission_image1 = add_poisson_noise(image1)
    gaussian_image1 = add_gaussian_noise(image1)
    salt_image1 = add_salt_and_pepper_noise(image1)
    poission_image2 = add_poisson_noise(image2)
    gaussian_image2 = add_gaussian_noise(image2)
    salt_image2 = add_salt_and_pepper_noise(image2)
    poission_image3 = add_poisson_noise(image3)
    gaussian_image3 = add_gaussian_noise(image3)
    salt_image3 = add_salt_and_pepper_noise(image3)
    cv2.imwrite('./static/poission_image1.png',poission_image1)
    cv2.imwrite('./static/gaussian_image1.png', gaussian_image1)
    cv2.imwrite('./static/salt_image1.png', salt_image1)
    cv2.imwrite('./static/poission_image2.png',poission_image2)
    cv2.imwrite('./static/gaussian_image2.png', gaussian_image2)
    cv2.imwrite('./static/salt_image2.png', salt_image2)
    cv2.imwrite('./static/poission_image3.png',poission_image3)
    cv2.imwrite('./static/gaussian_image3.png', gaussian_image3)
    cv2.imwrite('./static/salt_image3.png', salt_image3)

    # show_2(image1,poission_image,"poisson")
    # show_2(image2, gaussian_image, "gaussian")
    # show_2(image3, salt_image, "salt")






if __name__ == "__main__":
    image_1 = cv2.imread("./datas/cat.jpg")
    image_2 = cv2.imread("./datas/flower.jpg")
    image_3 = cv2.imread("./datas/sun.jpg")
    add_noisy(image_1,image_2,image_3)




