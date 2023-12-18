import os
import cv2
import numpy as np
def load_segmentation_images_and_extract_grayscale_sorted(folder_path):
    unique_grayscale_values = set()
    for filename in os.listdir(folder_path):

        if filename.endswith('.png') or filename.endswith('.jpg'):  # 假設影像檔案是PNG或JPG格式
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, 0)

            unique_grayscale_values |= set(np.unique(img))
            print(f"已讀取影像並提取灰階值: {filename}")  # 顯示進度
    return sorted(unique_grayscale_values)


def display_images_with_specific_grayscale(folder_path, specific_value):
    for filename in os.listdir(folder_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):  # 假設影像檔案是PNG或JPG格式
            img_path = os.path.join(folder_path, filename)
            img_rgb_path = os.path.join('annotations', filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_rgb = cv2.imread(img_rgb_path)
            if specific_value in np.unique(img):
                print(filename)
                cv2.imshow(f"影像包含灰階值 {specific_value} - {filename}", img_rgb)
                cv2.waitKey(0)  # 等待直到按下任意鍵
                cv2.destroyAllWindows()

# 使用範例
folder_path = './annotations_gray'
colors  = load_segmentation_images_and_extract_grayscale_sorted(folder_path)
print("所有出現過的不重複顏色:")

for color in colors:
    print(color)

# specific_value = 146 # 你想尋找的特定灰階值
# display_images_with_specific_grayscale(folder_path, specific_value)