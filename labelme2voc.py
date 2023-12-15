#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import os
import os.path as osp
import sys

import imgviz
import numpy as np
from os import listdir
import labelme
from pandas import read_csv
import PIL.Image
from PIL import Image
import io

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_dir", default="./json", type=str, help="input annotated directory")
    parser.add_argument("--output_dir", default="./", type=str, help="output dataset directory")
    parser.add_argument("--labels", default="./label.txt", type=str, help="labels file")
    parser.add_argument("--noviz", help="no visualization", action="store_true")

    return parser.parse_args()


def get_palette():
    df = read_csv('./color_coding_semantic_segmentation_classes - Sheet1.csv',)
    data = df.values
    palette = np.zeros((152, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]  # 将第一行设置为白色

    # palette = np.random.randint(0, 150, (150, 3), dtype=np.uint8)
    row_data_list = []
    # print(palette.shape)
    for i in range(150):
        palette[i+1][0]=int(data[i][5][1:-1].split(",")[2])
        palette[i+1][1]=int(data[i][5][1:-1].split(",")[1])
        palette[i+1][2]=int(data[i][5][1:-1].split(",")[0])
        # print(palette[i])
    row_data = data[:, 8]
    for item in row_data:
        row_data_list.append(str(item))
    return palette, row_data_list

def create_directories(input_dir, output_dir, create_viz=False):
    ## create path
    if osp.exists(input_dir):
        print("Output directory already exists:", input_dir)
    else:   
        print("Create input directory:", input_dir)
        os.makedirs(input_dir)
    if osp.exists(output_dir):
        print("Output directory already exists:", output_dir)
    else:   
        print("Create output directory:", output_dir)
        os.makedirs(output_dir)

    if osp.exists(osp.join(output_dir, "val_label_size")):
        print("val_label_size directory already exists:", osp.join(output_dir, "val_label_size"))
    else:   
        print("Create val_label_size directory:", osp.join(output_dir, "val_label_size"))
        os.makedirs(osp.join(output_dir, "val_label_size"))

    if osp.exists(osp.join(output_dir, "SegmentationClass")):
        print("SegmentationClass directory already exists:", osp.join(output_dir, "SegmentationClass"))
    else:   
        print("Create directory:", osp.join(output_dir, "SegmentationClass"))
        os.makedirs(osp.join(output_dir, "SegmentationClass"))


    if osp.exists(osp.join(output_dir, "annotations")):
        print("Output directory already exists:", osp.join(output_dir, "annotations"))
    else:   
        print("Create directory:",osp.join(output_dir, "annotations"))
        os.makedirs(osp.join(output_dir, "annotations"))
    if not create_viz:
        if osp.exists(osp.join(output_dir, "SegmentationClassVisualization")):
            print("Output directory already exists:",output_dir, "SegmentationClassVisualization")
        else:   
            print("Create directory:", osp.join(output_dir, "SegmentationClassVisualization"))
            os.makedirs(osp.join(output_dir, "SegmentationClassVisualization"))


def lblsave(filename, lbl,colormap):

    if osp.splitext(filename)[1] != ".png":
        filename += ".png"
    # Assume label ranses [-1, 254] for int32,
    # and [0, 255] for uint8 as VOC.
    if lbl.min() >= -1 and lbl.max() < 255:
        lbl_pil = PIL.Image.fromarray(lbl.astype(np.uint8), mode="P")
        
        
        lbl_pil.putpalette(colormap.flatten())
        lbl_pil.save(filename)
    else:
        raise ValueError(
            "[%s] Cannot save the pixel-wise class label as PNG. "
            "Please consider using the .npy format." % filename
        )

def shapes_to_label(img_shape, shapes, label_name_to_value):
    # background id == 0 
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []
    for shape in shapes:
        points = shape["points"]
        label = shape["label"]
        group_id = shape.get("group_id")
        # if group_id is None:
        #     group_id = uuid.uuid1()
        shape_type = shape.get("shape_type", None)

        cls_name = label
        # print(cls_name)
        instance = (cls_name, group_id)

        if instance not in instances:
            instances.append(instance)
        ins_id = instances.index(instance) + 1
        # print(label_name_to_value)
        cls_id = label_name_to_value[cls_name]

        mask = shape_to_mask(img_shape[:2], points, shape_type)
        cls[mask] = cls_id
        ins[mask] = ins_id
        # print(cls)
    return cls, ins

def shape_to_mask(img_shape, points, shape_type=None, line_width=10, point_size=5):

    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask






def main():
    args = parse_arguments()
    create_directories( args.input_dir, args.output_dir, create_viz=args.noviz)
    class_names = []
    class_name_to_id = {}    
    for i in range(151):
        class_names.append(str(i))  
    class_names = tuple(class_names)

    colormap,_ = get_palette()
    colormap = colormap[:, [2, 1, 0]]

    for i in range(151):
        class_name_to_id[str(i)] = i 

    i=0
    for filename in glob.glob(osp.join(args.input_dir, "*.json")):
        print("Generating dataset from:", filename)
        # i+=1
        # if i==2:
        #     break       

        # Read json
        label_file = labelme.LabelFile(filename=filename)      
        base = osp.splitext(osp.basename(filename))[0]    
    
        # org Images
        with open(osp.join(args.output_dir, "val_label_size", base + ".jpg"), "wb") as f:
            f.write(label_file.imageData)

        # SegmentationClass
        img = labelme.utils.img_data_to_arr(label_file.imageData)

        lbl, _ = shapes_to_label(
            img_shape=img.shape,
            shapes=label_file.shapes,
            label_name_to_value=class_name_to_id,
        )
        
        lblsave(osp.join(args.output_dir, "annotations", base + ".png"), lbl,colormap)

        # Numpy
        np.save(osp.join(args.output_dir, "SegmentationClass", base + ".npy"), lbl)

        # SegmentationClassVisualization
        if not args.noviz:
            viz = imgviz.label2rgb(
                lbl,
                imgviz.rgb2gray(img),
                font_size=15,
                label_names=class_names,
                loc="rb",
                colormap=colormap  # 使用自定義的顏色映射表
            )
            imgviz.io.imsave(osp.join(args.output_dir,"SegmentationClassVisualization",base + ".jpg",), viz)

if __name__ == "__main__":
    main()
