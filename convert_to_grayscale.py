import numpy as np
from PIL import Image
import cv2
import os
from pandas import read_csv
def get_palette():
    df = read_csv('./color_coding_semantic_segmentation_classes - Sheet1.csv',)
    data = df.values
    
    palette = np.zeros((151, 3), dtype=np.uint8)
    row_data_list = []

    palette[0] = [0, 0, 0]
    # print(palette.shape)
    for i in range(1, 151):
        palette[i][0]=int(data[i-1][5][1:-1].split(",")[2])
        palette[i][1]=int(data[i-1][5][1:-1].split(",")[1])
        palette[i][2]=int(data[i-1][5][1:-1].split(",")[0])
        # print(palette[i])
    row_data = data[:, 8]
    for item in row_data:
        row_data_list.append(str(item))
    # print(palette)
    return palette, row_data_list

def convert_to_grayscale(label, palette):
    # 初始化一個灰階圖片
    grayscale = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
    
    # 對於每個可能的顏色，找到對應的灰階值
    for gray_value, color in enumerate(palette):
        mask = np.all(label == color, axis=-1)
        grayscale[mask] = gray_value

    return grayscale

def process_all_images(input_dir, output_dir, palette, size=(128, 128)):
    # 確保輸出目錄存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍歷輸入目錄中的所有圖片
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):  # 確保處理的是圖片文件
            file_path = os.path.join(input_dir, filename)
            label = cv2.imread(file_path, cv2.IMREAD_COLOR)

            # 轉換成灰階
            grayscale_image = convert_to_grayscale(label, palette)

            # 將 Numpy 數組轉換為 PIL Image 對象
            pil_img = Image.fromarray(grayscale_image)

            # 調整圖片大小
            resized_img = pil_img.resize(size, Image.ANTIALIAS)

            # 構建輸出文件的路徑
            output_file_path = os.path.join(output_dir, filename)

            # 保存轉換後的圖片
            resized_img.save(output_file_path)
            print(f"圖片已保存: {output_file_path}")

if __name__ == "__main__":
    input_dir = './annotations'
    output_dir = './annotations_gray'
    palette = get_palette()[0]

    process_all_images(input_dir, output_dir, palette,size=(960, 540))
