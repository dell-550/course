import cv2
import numpy as np

# 读取多张图片 salt gaussian poisson


data = ["salt", "gaussian", "poisson"]


def show_noisy(datas):
    for data in datas:
        source_image = cv2.imread(r'E:\python\image_spatial_filtering\static\data_1\source_source.png',
                                  cv2.IMREAD_GRAYSCALE)
        noisy = cv2.imread(f'E:\python\image_spatial_filtering\static\data_1\{data}_source.jpg', cv2.IMREAD_GRAYSCALE)

        # 创建一个灰度图像的白色空白区域
        blank_space = np.zeros((source_image.shape[0], 50), dtype=np.uint8)
        blank_space.fill(255)  # 白色填充

        # 水平拼接图片和空白区域
        combined_image = np.hstack((source_image, blank_space, noisy))

        # 显示拼接后的图像
        # cv2.imshow('Combined Image', combined_image)

        # 保存拼接后的图像
        cv2.imwrite(f'./static/{data}.png', combined_image)

        # cv2.waitKey(0)  # 等待用户按键或关闭窗口
        #
        # # 关闭窗口
        # cv2.destroyAllWindows()


show_noisy(data)
