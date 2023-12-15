# Labelme
- [Labelme tutorial](https://hackmd.io/@agbug/ByjZArc9h)
- [Labelme github](https://github.com/wkentaro/labelme)
## Preparation
### Prepare Folder
1. [Download ICALAB_val_image](https://drive.google.com/drive/folders/1XAJ9Pcwi1-pNGjYvxFnvrbDn38dt1zP4?usp=drive_link)
![](https://hackmd.io/_uploads/rkJl2dsc3.png)


3. Clone [labelme_tutorial](https://github.com/patrick0115/labelme_tutorial) or download ZIP
```
git clone git@github.com:patrick0115/labelme_tutorial.git
```
- Folder structure
```bash
Anypath
├── ICALAB_val_image
│   ├── JPEGImages
│   ├── SegmentationClass 
│   ├── json  
│   ├── SegmentationClassPNG
│   ├── SegmentationClassVisualization  
│   ├── color_coding_semantic_segmentation_classes - Sheet1.csv
│   ├── class_names.txt
│   ├── images  
│   │    │── 0000.jpg
│   │    │── 0001.jpg
│   │    │── 0002.jpg
│   │    │ .......
├── labelme_tutorial 
│   ├── json
│   ├── output
│   ├── label.txt
```
![](https://hackmd.io/_uploads/HyAq4K55n.png)
![](https://hackmd.io/_uploads/r1RD6wc53.png)

### Install labelme
```bash
pip install labelme
```
### Open labelme
```bash
labelme
```
![](https://hackmd.io/_uploads/HyJ7RIq9n.png)
## Start label
1. Open Dir
Choose images folder with raw images.
![](https://hackmd.io/_uploads/B1ZqR8q93.png)
2. Label
- 描繪
![](https://hackmd.io/_uploads/Syz1eP9qh.png)
- 編輯圖案
可以移動複製,調整點位
![](https://hackmd.io/_uploads/H1-gePcch.png)
![](https://hackmd.io/_uploads/HJoGePq52.png)
- 修改類別
![](https://hackmd.io/_uploads/B18Hxwq53.png)
- 畫面色調調整
若圖片不清楚，可以調整圖案亮度
![](https://hackmd.io/_uploads/rysTlv5c2.png)

- 儲存在 labelme_tutorial/json
![](https://hackmd.io/_uploads/ByJArYc52.png)

## Start Semantic Segmentation
```bash
python .\labelme2voc.py
```
- 會在output資料夾產生一下檔案

![](https://hackmd.io/_uploads/Skw3195c3.png)

- 上傳到對應資料夾(包含json檔)
![](https://hackmd.io/_uploads/BJhAjOs93.png)


