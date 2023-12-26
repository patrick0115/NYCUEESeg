import os
import cv2
import numpy as np
from collections import defaultdict
def read_image_grayscale(img_path):
    """ 讀取指定路徑的灰階影像。"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"無法讀取影像: {img_path}")
    return img

def count_grayscale_classes_appearance(folder_path):
    """ 計算每個灰階類別在不同影像中出現的次數，並分別印出每個類別的次數以及出現次數最多的前10名類別。"""
    class_count = defaultdict(int)
    errors = []

    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg')):
            try:
                img = read_image_grayscale(os.path.join(folder_path, filename))
                unique = np.unique(img)
                for u in unique:
                    class_count[u] += 1
            except IOError as e:
                errors.append(str(e))

    # 首先印出每個類別的出現次數
    for cls, count in class_count.items():
        print(f"類別 {cls} 在不同的影像中出現的次數: {count}")

    # 將字典轉換為元組列表並按出現次數降序排序
    sorted_class_count = sorted(class_count.items(), key=lambda item: item[1], reverse=True)

    # 印出出現次數最多的前10名類別
    print("\n出現次數最多的前11名類別:")
    for cls, count in sorted_class_count[:11]:
        print(f"類別 {cls} 在不同的影像中出現的次數: {count}")

    if errors:
        print("\n以下影像無法讀取:")
        for error in errors:
            print(error)



def count_grayscale_classes_pixel_appearance(folder_path):
    class_count = defaultdict(int)

    for filename in os.listdir(folder_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                unique, counts = np.unique(img, return_counts=True)
                for u, c in zip(unique, counts):
                    class_count[u] += c
            else:
                print(f"無法讀取影像: {filename}")
    for cls, count in class_count.items():
        print(f"類別 {cls} 在不同的影像中出現的次數: {count}")


def load_segmentation_images_and_extract_grayscale_sorted(folder_path):
    """ 讀取分割影像並提取灰階值，並排序。"""
    unique_grayscale_values = set()
    errors = []
    num_classes_per_image = []

    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg')):
            try:
                img = read_image_grayscale(os.path.join(folder_path, filename))
                unique_classes = np.unique(img)
                unique_grayscale_values.update(unique_classes)
                num_classes_per_image.append(len(unique_classes))
                print(f"影像 '{filename}' 包含以下類別 {unique_classes}") 
            except IOError as e:
                errors.append(str(e))

    average_classes = np.mean(num_classes_per_image)
    print(f"平均每張影像類別數量: {average_classes}")
    print(f"所有出現過的不重複灰階值: {sorted(unique_grayscale_values)}")
    
    if errors:
        print("以下影像無法讀取:")
        for error in errors:
            print(error)

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
if __name__ == "__main__":
    # 使用範例
    folder_path = './annotations_gray'
    # load_segmentation_images_and_extract_grayscale_sorted(folder_path)

    # count_grayscale_classes_pixel_appearance(folder_path)

    count_grayscale_classes_appearance(folder_path)

    # display_images_with_specific_grayscale(folder_path, specific_value = 146 )