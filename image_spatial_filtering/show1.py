import cv2
from matplotlib import pyplot as plt

# average  laplacian median sharpen
data = ["average", "laplacian", "median", "sharpen"]


def show_filter(datas):
    for data in datas:
        source_image = cv2.imread(r'E:\python\image_spatial_filtering\static\data_1\source_source.png',
                                  cv2.IMREAD_GRAYSCALE)
        source_average = cv2.imread(f'E:\python\image_spatial_filtering\static\data_1\source_{data}_self.png',
                                    cv2.IMREAD_GRAYSCALE)
        source_image1 = cv2.imread(r'E:\python\image_spatial_filtering\static\data_2\source_source.png',
                                   cv2.IMREAD_GRAYSCALE)
        source_average1 = cv2.imread(f'E:\python\image_spatial_filtering\static\data_2\source_{data}_self.png',
                                     cv2.IMREAD_GRAYSCALE)
        source_image2 = cv2.imread(r'E:\python\image_spatial_filtering\static\data_3\source_source.png',
                                   cv2.IMREAD_GRAYSCALE)
        source_average2 = cv2.imread(f'E:\python\image_spatial_filtering\static\data_3\source_{data}_self.png',
                                     cv2.IMREAD_GRAYSCALE)
        plt.subplot(2, 3, 1)
        plt.title('source_image')
        plt.imshow(source_image, cmap='gray')
        plt.axis('off')
        plt.subplot(2, 3, 4)
        plt.title(f'{data}_image')
        plt.imshow(source_average, cmap='gray')
        plt.axis('off')
        plt.subplot(2, 3, 2)
        plt.title('source_image')
        plt.imshow(source_image1, cmap='gray')
        plt.axis('off')
        plt.subplot(2, 3, 5)
        plt.title(f'{data}_image')
        plt.imshow(source_average1, cmap='gray')
        plt.axis('off')
        plt.subplot(2, 3, 3)
        plt.title('source_image')
        plt.imshow(source_image2, cmap='gray')
        plt.axis('off')
        plt.subplot(2, 3, 6)
        plt.title(f'{data}_image')
        plt.imshow(source_average2, cmap='gray')
        plt.axis('off')
        plt.savefig(f"./static/source_{data}.png")
        plt.show()


show_filter(data)
