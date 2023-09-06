# @Author: Pieter Blok
# @Date:   2021-03-26 14:30:31
# @Last Modified by:   Pieter Blok
# @Last Modified time: 2023-03-17 12:36:09

import random
import os
import numpy as np
from PIL import Image
import shutil
import cv2
import json
import math
import datetime
import time
from tqdm import tqdm
import itertools

supported_cv2_formats = (".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tiff", ".tif")

def find_valid_images_and_annotations(rootdir):
    image_annotation_pairs = []

    for root, dirs, files in tqdm(list(os.walk(rootdir))):
        for name in files:
            if name.endswith((".json")):
                full_path = os.path.join(root, name)
                with open(full_path, 'r') as json_file:
                    try:
                        data = json.load(json_file)

                        ## labelme
                        if 'version' in data:
                            annot_format = 'labelme'
                            if 'shapes' in data:
                                if len(data['shapes']) > 0:
                                    imgname = data['imagePath']
                                    imgpath = os.path.join(root, imgname)

                                    if os.path.exists(imgpath):
                                        image_annotation_pairs.append([imgpath, full_path, annot_format])

                        ## v7-darwin                
                        if 'darwin' in data['image']['url']:
                            annot_format = 'darwin'
                            if 'annotations' in data:
                                if len(data['annotations']) > 0:
                                    imgname = data['image']['original_filename']
                                    imgpath = os.path.join(root, imgname)

                                    if os.path.exists(imgpath):
                                        image_annotation_pairs.append([imgpath, full_path, annot_format])

                    except:
                        continue
                
    return image_annotation_pairs


def process_labelme_json(jsonfile, classnames):
    group_ids = []

    with open(jsonfile, 'r') as json_file:
        data = json.load(json_file)
        for p in data['shapes']:
            group_ids.append(p['group_id'])

    only_group_ids = [x for x in group_ids if x is not None]
    unique_group_ids = list(set(only_group_ids))
    no_group_ids = sum(x is None for x in group_ids)
    total_masks = len(unique_group_ids) + no_group_ids

    all_unique_masks = np.zeros(total_masks, dtype = object)

    if len(unique_group_ids) > 0:
        unique_group_ids.sort()

        for k in range(len(unique_group_ids)):
            unique_group_id = unique_group_ids[k]
            all_unique_masks[k] = unique_group_id

        for h in range(no_group_ids):
            all_unique_masks[len(unique_group_ids) + h] = "None" + str(h+1)
    else:
        for h in range(no_group_ids):
            all_unique_masks[h] = "None" + str(h+1)    

    category_ids = []
    masks = []
    crowd_ids = []

    for i in range(total_masks):
        category_ids.append([])
        masks.append([])
        crowd_ids.append([])

    none_counter = 0 

    for p in data['shapes']:
        group_id = p['group_id']

        if group_id is None:
            none_counter = none_counter + 1
            fill_id = int(np.where(np.asarray(all_unique_masks) == (str(group_id) + str(none_counter)))[0][0])
        else:
            fill_id = int(np.where(np.asarray(all_unique_masks) == group_id)[0][0])

        classname = p['label']

        try:
            category_id = int(np.where(np.asarray(classnames) == classname)[0][0] + 1)
            category_ids[fill_id] = category_id
            run_further = True
        except:
            print("Cannot find the class name (please check the annotation files)")
            run_further = False

        if run_further:
            if p['shape_type'] == "circle":
                # https://github.com/wkentaro/labelme/issues/537
                bearing_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 
                180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345, 360]
                            
                orig_x1 = p['points'][0][0]
                orig_y1 = p['points'][0][1]

                orig_x2 = p['points'][1][0]
                orig_y2 = p['points'][1][1]

                cx = (orig_x2 - orig_x1)**2
                cy = (orig_y2 - orig_y1)**2
                radius = math.sqrt(cx + cy)

                circle_polygon = []
            
                for k in range(0, len(bearing_angles) - 1):
                    ad1 = math.radians(bearing_angles[k])
                    x1 = radius * math.cos(ad1)
                    y1 = radius * math.sin(ad1)
                    circle_polygon.append( (orig_x1 + x1, orig_y1 + y1) )

                    ad2 = math.radians(bearing_angles[k+1])
                    x2 = radius * math.cos(ad2)  
                    y2 = radius * math.sin(ad2)
                    circle_polygon.append( (orig_x1 + x2, orig_y1 + y2) )

                pts = np.asarray(circle_polygon).astype(np.float32)
                pts = pts.reshape((-1,1,2))
                points = np.asarray(pts).flatten().tolist()
                
            if p['shape_type'] == "rectangle":
                (x1, y1), (x2, y2) = p['points']
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                points = [x1, y1, x2, y1, x2, y2, x1, y2]

            if p['shape_type'] == "polygon":
                points = p['points']
                pts = np.asarray(points).astype(np.float32).reshape(-1,1,2)   
                points = np.asarray(pts).flatten().tolist()

            masks[fill_id].append(points)

            ## labelme version 4.5.6 does not have a crowd_id, so fill it with zeros
            crowd_ids[fill_id] = 0
            status = "successful"
        else:
            status = "unsuccessful"

    return category_ids, masks, crowd_ids, status


def process_darwin_json(jsonfile, classnames):
    
    with open(jsonfile, 'r') as json_file:
        data = json.load(json_file)

    total_masks = len(data['annotations'])
    category_ids = []
    masks = []
    crowd_ids = []

    for i in range(total_masks):
        category_ids.append([])
        masks.append([])
        crowd_ids.append([])

    fill_id = 0 

    for p in data['annotations']:
        classname = p['name']

        try:
            category_id = int(np.where(np.asarray(classnames) == classname)[0][0] + 1)
            category_ids[fill_id] = category_id
            run_further = True
        except:
            print("Cannot find the class name (please check the annotation files)")
            run_further = False

        if run_further:
            if 'polygon' in p:
                if 'path' in p['polygon']:
                    points = []
                    path_points = p['polygon']['path']
                    for h in range(len(path_points)):
                        points.append(path_points[h]['x'])
                        points.append(path_points[h]['y'])

                    masks[fill_id].append(points)

            if 'complex_polygon' in p:
                if 'path' in p['complex_polygon']:
                    for k in range(len(p['complex_polygon']['path'])):
                        points = []
                        path_points = p['complex_polygon']['path'][k]
                        for h in range(len(path_points)):
                            points.append(path_points[h]['x'])
                            points.append(path_points[h]['y'])

                        masks[fill_id].append(points)
                    
            crowd_ids[fill_id] = 0
            status = "successful"
        else:
            status = "unsuccessful"

        fill_id += 1

    return category_ids, masks, crowd_ids, status


def bounding_box(masks):
    areas = []
    boxes = []

    for _ in range(len(masks)):
        areas.append([])
        boxes.append([])


    for i in range(len(masks)):
        points = masks[i]
        all_points = np.concatenate(points)

        pts = np.asarray(all_points).astype(np.float32).reshape(-1,1,2)
        bbx,bby,bbw,bbh = cv2.boundingRect(pts)

        area = bbw*bbh 
        areas[i] = area                      
        boxes[i] = [bbx,bby,bbw,bbh]

    return areas, boxes


def visualize_annotations(img, category_ids, masks, boxes, classes):
    colors = [(0, 255, 0), (255, 0, 0), (255, 0, 255), (0, 0, 255), (0, 255, 255), (255, 255, 255)]
    color_list = np.remainder(np.arange(len(classes)), len(colors))
    
    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1
    font_thickness = 1
    thickness = 3
    text_color1 = [255, 255, 255]
    text_color2 = [0, 0, 0]

    img_vis = img.copy()

    for i in range(len(masks)):
        points = masks[i]
        bbx,bby,bbw,bbh = boxes[i]
        category_id = category_ids[i]
        class_id = category_id-1
        _class = classes[class_id]
        color = colors[color_list[class_id]]

        for j in range(len(points)):
            point_set = points[j]
            pntset = np.asarray(point_set).astype(np.int32).reshape(-1,1,2) 
            img_vis = cv2.polylines(img_vis, [pntset], True, color, thickness)

        img_vis = cv2.rectangle(img_vis, (bbx, bby), ((bbx+bbw), (bby+bbh)), color, thickness)

        text_str = "{:s}".format(_class)
        text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

        if bby < 100:
            text_pt = (bbx, bby+bbh)
        else:
            text_pt = (bbx, bby)

        img_vis = cv2.rectangle(img_vis, (text_pt[0], text_pt[1] + 7), (text_pt[0] + text_w, text_pt[1] - text_h - 7), text_color1, -1)
        img_vis = cv2.putText(img_vis, text_str, (text_pt[0], text_pt[1]), font_face, font_scale, text_color2, font_thickness, cv2.LINE_AA)

    return img_vis


def visualize_mrcnn(img_np, classes, scores, masks, boxes, class_names):
    masks = masks.astype(dtype=np.uint8)
    font_scale = 0.6
    font_thickness = 1
    text_color = [0, 0, 0]

    if masks.any():
        maskstransposed = masks.transpose(1,2,0) # transform the mask in the same format as the input image array (h,w,num_dets)
        red_mask = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1]),dtype=np.uint8)
        blue_mask = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1]),dtype=np.uint8)
        green_mask = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1]),dtype=np.uint8)
        all_masks = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1],3),dtype=np.uint8) # BGR

        colors = [(0, 255, 0), (255, 0, 0), (255, 0, 255), (0, 0, 255), (0, 255, 255), (255, 255, 255)]
        color_list = np.remainder(np.arange(len(class_names)), len(colors))
        imgcopy = img_np.copy()

        for i in range (maskstransposed.shape[-1]):
            color = colors[color_list[classes[i]]]
            x1, y1, x2, y2 = boxes[i, :]
            cv2.rectangle(imgcopy, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

            _class = class_names[classes[i]]
            text_str = '%s: %.2f' % (_class, scores[i])

            font_face = cv2.FONT_HERSHEY_DUPLEX

            text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
            text_pt = (int(x1), int(y1) - 3)

            cv2.rectangle(imgcopy, (int(x1), int(y1)), (int(x1) + text_w, int(y1) - text_h - 4), color, -1)
            cv2.putText(imgcopy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

            mask = maskstransposed[:,:,i]

            if colors.index(color) == 0: # green
                green_mask = cv2.add(green_mask,mask)
            elif colors.index(color) == 1: # blue
                blue_mask = cv2.add(blue_mask,mask)
            elif colors.index(color) == 2: # magenta
                blue_mask = cv2.add(blue_mask,mask)
                red_mask = cv2.add(red_mask,mask)
            elif colors.index(color) == 3: # red
                red_mask = cv2.add(red_mask,mask)
            elif colors.index(color) == 4: # yellow
                green_mask = cv2.add(green_mask,mask)
                red_mask = cv2.add(red_mask,mask)
            else: #white
                blue_mask = cv2.add(blue_mask,mask)
                green_mask = cv2.add(green_mask,mask)
                red_mask = cv2.add(red_mask,mask)

        all_masks[:,:,0] = blue_mask
        all_masks[:,:,1] = green_mask
        all_masks[:,:,2] = red_mask
        all_masks = np.multiply(all_masks,255).astype(np.uint8)

        img_mask = cv2.addWeighted(imgcopy,1,all_masks,0.5,0)
    else:
        img_mask = img_np

    return img_mask


def write_file(imgdir, images, name):
    with open(os.path.join(imgdir, "{:s}.txt".format(name)), 'w') as f:
        for img in images:
            f.write("{:s}\n".format(img))


def split_datasets_randomly(rootdir, images, train_val_test_split):
    all_ids = np.arange(len(images))
    random.shuffle(all_ids)

    train_slice = int(train_val_test_split[0]*len(images))
    val_slice = int(train_val_test_split[1]*len(images))

    train_ids = all_ids[:train_slice]
    val_ids = all_ids[train_slice:train_slice+val_slice]
    test_ids = all_ids[train_slice+val_slice:]

    train_images = np.array(images)[train_ids].tolist()
    val_images = np.array(images)[val_ids].tolist()
    test_images = np.array(images)[test_ids].tolist()

    train_image_names = [train_images[t][0] for t in range(len(train_images))]
    val_image_names = [val_images[v][0] for v in range(len(val_images))]
    test_image_names = [test_images[te][0] for te in range(len(test_images))]

    write_file(rootdir, train_image_names, "train")
    write_file(rootdir, val_image_names, "val")
    write_file(rootdir, test_image_names, "test") 
    
    return [train_images, val_images, test_images], ["train", "val", "test"]


def create_json(rootdir, img_annot, classes, name):
    date_created = datetime.datetime.now()
    year_created = date_created.year

    ## initialize the final json file
    writedata = {}
    writedata['info'] = {"description": "description", "url": "url", "version": str(1), "year": str(year_created), "contributor": "contributor", "date_created": str(date_created)}
    writedata['licenses'] = []
    writedata['licenses'].append({"url": "license_url", "id": "license_id", "name": "license_name"})
    writedata['images'] = []
    writedata['type'] = "instances"
    writedata['annotations'] = []
    writedata['categories'] = []

    for k in range(len(classes)):
        superclass = classes[k]
        writedata['categories'].append({"supercategory": superclass, "id": (k+1), "name": superclass})

    annotation_id = 1   ## see: https://github.com/cocodataset/cocoapi/issues/507
    output_file = name + ".json"

    print("")
    print(output_file)

    for j in tqdm(range(len(img_annot))):
        imgname = img_annot[j][0]
        annotname = img_annot[j][1]
        annot_format = img_annot[j][2]
        
        img = cv2.imread(imgname)
        height, width, channels = img.shape

        imgpathname = imgname.replace(rootdir+'/', '')

        try:
            modTimesinceEpoc = os.path.getmtime(imgname)
            modificationTime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modTimesinceEpoc))
            date_modified = modificationTime
        except:
            date_modified = None
        
        writedata['images'].append({
                        'license': 0,
                        'url': None,
                        'file_name': imgpathname,
                        'height': height,
                        'width': width,
                        'date_captured': None,
                        'id': j
                    })

        # Procedure to store the annotations in the final JSON file
        if annot_format == 'labelme':
            category_ids, masks, crowd_ids, status = process_labelme_json(annotname, classes)
            
        if annot_format == 'darwin':
            category_ids, masks, crowd_ids, status = process_darwin_json(annotname, classes)

        areas, boxes = bounding_box(masks)
        img_vis = visualize_annotations(img, category_ids, masks, boxes, classes)

        for q in range(len(category_ids)):
            category_id = category_ids[q]
            mask = masks[q]
            bb_area = areas[q]
            bbpoints = boxes[q]
            crowd_id = crowd_ids[q]

            writedata['annotations'].append({
                    'id': annotation_id,
                    'image_id': j,
                    'category_id': category_id,
                    'segmentation': mask,
                    'area': bb_area,
                    'bbox': bbpoints,
                    'iscrowd': crowd_id
                })
    
            annotation_id = annotation_id+1
            
    with open(os.path.join(rootdir, output_file), 'w') as outfile:
        json.dump(writedata, outfile)


def get_class_names(rootdir, img_annot):
    unique_class_names = []

    for j in tqdm(range(len(img_annot))):
        imgname = img_annot[j][0]
        annotname = img_annot[j][1]
        annot_format = img_annot[j][2]

        # Procedure to store the annotations in the final JSON file
        if annot_format == 'labelme':
            with open(annotname, 'r') as json_file:
                data = json.load(json_file)
            for p in data['shapes']:
                classname = p['label']
                if classname not in unique_class_names:
                    unique_class_names.append(classname)
            
        if annot_format == 'darwin':
            with open(annotname, 'r') as json_file:
                data = json.load(json_file)
            for p in data['annotations']:
                classname = p['name']
                if classname not in unique_class_names:
                    unique_class_names.append(classname)

    return unique_class_names


## the function below is heavily inspired by the function "repeat_factors_from_category_frequency" in ./detectron2/data/samplers/distributed_sampler.py
def calculate_repeat_threshold(classes, minority_classes, repeat_factor_smallest_class, dataset_dicts_train):
    images_with_class_annotations = np.zeros(len(classes)).astype(np.int16)
    for d in range(len(dataset_dicts_train)):
        data_point = dataset_dicts_train[d]
        classes_annot = []
        for k in range(len(data_point["annotations"])):
            classes_annot.append(data_point["annotations"][k]['category_id'])
        unique_classes = list(set(classes_annot))
        for c in unique_classes:
            images_with_class_annotations[c] += 1

    for mc in range(len(minority_classes)):
        minorty_class = minority_classes[mc]
        search_id = classes.index(minorty_class)
        image_count = images_with_class_annotations[search_id]

        try:
            if image_count < min_value:
                min_value = image_count
        except:
            min_value = image_count
    
    repeat_threshold = np.power(repeat_factor_smallest_class, 2) * (min_value / len(dataset_dicts_train))
    repeat_threshold = np.clip(repeat_threshold, 0, 1)
    return float(repeat_threshold)


def find_class_names(rootdir):
    image_annotation_pairs = find_valid_images_and_annotations(rootdir)
    print("{:d} valid images and annotations found!".format(len(image_annotation_pairs)))
    class_names = get_class_names(rootdir, image_annotation_pairs)
    return class_names


## the function below is heavily inspired by the function "print_instances_class_histogram" in ./detectron2/data/build.py
def get_minority_classes(dataset_dicts, class_names, percentage):
    """
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,), dtype=np.int)
    for entry in dataset_dicts:
        annos = entry["annotations"]
        classes = np.asarray(
            [x["category_id"] for x in annos if not x.get("iscrowd", 0)], dtype=np.int
        )
        if len(classes):
            assert classes.min() >= 0, f"Got an invalid category_id={classes.min()}"
            assert (
                classes.max() < num_classes
            ), f"Got an invalid category_id={classes.max()} for a dataset of {num_classes} classes"
        histogram += np.histogram(classes, bins=hist_bins)[0]

    N_COLS = min(6, len(class_names) * 2)


    data = list(
        itertools.chain(*[[class_names[i], int(v)] for i, v in enumerate(histogram)])
    )
    
    data_array = np.array(data).reshape(num_classes, 2)
    counts = data_array[:, 1].astype(np.int16)
    percs = np.array([counts[c]/sum(counts) for c in range(len(counts))]).astype(np.float32)

    minority_classes = []
    for c in range(num_classes):
        if percs[c] < percentage:
            minority_classes.append(data_array[c, 0])

    return minority_classes
    

def prepare_dataset(rootdir, classes, train_val_test_split):
    image_annotation_pairs = find_valid_images_and_annotations(rootdir)
    print("{:d} valid images and annotations found!".format(len(image_annotation_pairs)))

    print("Converting annotations...")
    datasets, names = split_datasets_randomly(rootdir, image_annotation_pairs, train_val_test_split)

    for dataset, name in zip(datasets, names):
        create_json(rootdir, dataset, classes, name)


def prepare_organized_dataset(rootdir, traindir, valdir, testdir, classes):
    for imgdir, name in zip([traindir, valdir, testdir], ['train', 'val', 'test']):
        print("Processing {:s} directory...".format(name))
        image_annotation_pairs = find_valid_images_and_annotations(imgdir)
        print("{:d} valid images and annotations found!".format(len(image_annotation_pairs)))

        print("Converting annotations...")
        create_json(rootdir, image_annotation_pairs, classes, name)


def prepare_organized_testset(rootdir, testdir, classes):
    print("Processing {:s} directory...".format('test'))
    image_annotation_pairs = find_valid_images_and_annotations(testdir)
    print("{:d} valid images and annotations found!".format(len(image_annotation_pairs)))

    print("Converting annotations...")
    create_json(rootdir, image_annotation_pairs, classes, 'test')