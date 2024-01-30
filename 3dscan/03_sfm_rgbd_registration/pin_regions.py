import os
import cv2
import open3d as o3d
from pycocotools.coco import COCO
import numpy as np
import pandas as pd
import copy
import json

class PinRegions:
    def __init__(self, img_root, coco_file, csv_file, intrinsics_file):
        self.img_root = img_root
        self.coco = COCO(coco_file)
        self.df = pd.read_csv(csv_file)
        self.intrinsics = self.load_intrinsics(intrinsics_file)


    def load_intrinsics(self, intrinsics_file):
        with open(intrinsics_file) as json_file:
            data = json.load(json_file)
        intrinsics = o3d.camera.PinholeCameraIntrinsic(data['width'], data['height'], data['intrinsic_matrix'][0], data['intrinsic_matrix'][4], data['intrinsic_matrix'][6], data['intrinsic_matrix'][7])

        return intrinsics


    def histogram_filtering(self, dimg, mask, max_depth_range=150, max_depth_contribution=0.05):
        mask = mask.astype(np.uint8)
        mask_bool = mask.astype(bool)
        
        z = np.expand_dims(dimg, axis=2)
        z_mask = z[mask_bool]
        z_mask_filtered = z_mask[z_mask != 0]

        if z_mask_filtered.size > 1: 
            z_mask_filtered_range = np.max(z_mask_filtered)-np.min(z_mask_filtered)

            if (z_mask_filtered_range > max_depth_range):
                hist, bin_edges = np.histogram(z_mask_filtered, density=False) 
                hist_peak = np.argmax(hist)
                lb = bin_edges[hist_peak]
                ub = bin_edges[hist_peak+1]

                bc = np.bincount(np.absolute(z_mask_filtered.astype(np.int64)))
                peak_id = np.argmax(bc)

                if peak_id > int(lb) and peak_id < int(ub):
                    peak_id = peak_id
                else:
                    bc_clip = bc[int(lb):int(ub)]
                    peak_id = int(lb) + np.argmax(bc_clip)

                pixel_counts = np.zeros((10), dtype=np.int64)

                for j in range(10):
                    lower_bound = peak_id-(max_depth_range - (j * 10))
                    upper_bound = lower_bound + max_depth_range
                    z_final = z_mask_filtered[np.where(np.logical_and(z_mask_filtered >= lower_bound, z_mask_filtered <= upper_bound))]
                    pixel_counts[j] = z_final.size

                pix_id = np.argmax(pixel_counts)
                lower_bound = peak_id-(max_depth_range - (pix_id * 10))
                upper_bound = lower_bound + max_depth_range
                z_final = z_mask_filtered[np.where(np.logical_and(z_mask_filtered >= lower_bound, z_mask_filtered <= upper_bound))]
                
            else:
                z_final = z_mask_filtered

            hist_f, bin_edges_f = np.histogram(z_final, density=False)
            norm1 = hist_f / np.sum(hist_f)

            sel1 = bin_edges_f[np.where(norm1 >= max_depth_contribution)]
            sel2 = bin_edges_f[np.where(norm1 >= max_depth_contribution)[0]+1]
            edges = np.concatenate((sel1,sel2), axis=0)
            final_bins = np.unique(edges)
    
            z_min = np.min(final_bins)
            z_max = np.max(final_bins)
        else:
            z_min = 0
            z_max = 0
        
        return z_min, z_max


    def binary_mask(self, img, mask):
        width, height, _ = img.shape
        mask_img = np.zeros((width, height)).astype(np.uint8)
        mask = np.array(mask, dtype=np.int32)
        plot_mask = mask.reshape(-1, 1, 2)
        cv2.fillPoly(mask_img, [plot_mask], 255)

        return mask_img


    def process_pcd(self, img, dimg, bin_potato, bin_pin, name, gt_depth, visualize=False):
        img_potato = np.multiply(img, np.expand_dims(bin_potato, axis=2))
        dimg_potato = np.multiply(dimg, bin_potato)
        dimg_potato_vis = dimg_potato.astype(np.uint8)

        z_min, z_max = self.histogram_filtering(dimg, bin_potato, gt_depth, 0.02)
        dimg_potato[dimg_potato < z_min] = 0
        dimg_potato[dimg_potato > z_max] = 0

        rgb_potato = o3d.geometry.Image((img_potato[:,:,::-1]).astype(np.uint8))
        depth_potato = o3d.geometry.Image(dimg_potato)
        rgbd_potato = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_potato, depth_potato, depth_scale=1000.0, depth_trunc=0.4, convert_rgb_to_intensity=False)
        pcd_potato = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_potato, self.intrinsics)

        img_pin = np.multiply(img, np.expand_dims(bin_pin, axis=2))
        dimg_pin = np.multiply(dimg, bin_pin)
        dimg_pin_vis = dimg_pin.astype(np.uint8)

        rgb_pin = o3d.geometry.Image((img_pin[:,:,::-1]).astype(np.uint8))
        depth_pin = o3d.geometry.Image(dimg_pin)
        rgbd_pin = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_pin, depth_pin, depth_scale=1000.0, depth_trunc=0.4, convert_rgb_to_intensity=False)
        pcd_pin = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_pin, self.intrinsics)
        pcd_pin.paint_uniform_color([1, 1, 0])

        if visualize:
            pcd_potato_vis = pcd_potato.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            pcd_pin_vis = pcd_pin.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            o3d.visualization.draw_geometries([pcd_potato_vis, pcd_pin_vis], window_name=f"{name} with pin region in yellow")

        return pcd_pin

    def draw_mask(self, img, category, bbox, mask, color):
        mask = np.array(mask, dtype=np.int32)
        plot_mask = mask.reshape(-1, 1, 2)

        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), color, 2)     
        cv2.putText(img, category, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.polylines(img, [plot_mask], True, color, 2) 


    def visualize_annotations(self, visualize_pcd=False):        
        img_ids = self.coco.getImgIds()
        img_infos = self.coco.loadImgs(img_ids)
        img_names = [img_info['file_name'] for img_info in img_infos]
        img_names.sort()

        for img_name in img_names:
            img_info = next(img_info for img_info in img_infos if img_info['file_name'] == img_name)

            img_path = os.path.join(self.img_root, img_name)
            potato_label = os.path.dirname(img_name)
            img_basename = os.path.splitext(os.path.basename(img_name))[0].replace("_rgb_", "_pcd_") 
            gt_depth = self.df.loc[self.df['label'] == potato_label, 'x3_depth_mm'].values[0]

            rgba = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img = rgba[:,:,:-1]
            mask = rgba[:,:,-1]
            img_vis = copy.deepcopy(img)

            dimg_name = img_path.replace("_rgb_", "_depth_")
            dimg_path = os.path.join(self.img_root, dimg_name)
            dimg = cv2.imread(dimg_path, cv2.IMREAD_UNCHANGED)
            
            img_id = img_info['id']
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            annotations = self.coco.loadAnns(ann_ids)

            for annotation in annotations:
                bbox = annotation['bbox']
                category_id = annotation['category_id']
                category = self.coco.loadCats(category_id)[0]['name']
                color = [0, 0, 255]
                
                if 'segmentation' in annotation:
                    pin_region = annotation['segmentation']
                    self.draw_mask(img_vis, category, bbox, pin_region, color)

                    if visualize_pcd:
                        pin_mask = self.binary_mask(img, pin_region)
                        bin_pin = pin_mask.astype(bool)
                        bin_mask = mask.astype(bool)
                        self.process_pcd(img, dimg, bin_mask, bin_pin, img_basename, gt_depth)

            cv2.imshow('Image with pin regions in red', img_vis)
            cv2.waitKey(1)

        cv2.destroyAllWindows()



if __name__ == '__main__':
    img_root = '/mnt/data/PieterBlok/Potato/Data/3DPotatoTwin/1_rgbd/1_image'
    coco_file = '/mnt/data/PieterBlok/Potato/Data/3DPotatoTwin/1_rgbd/pin_regions.json'
    csv_file = '/mnt/data/PieterBlok/Potato/Data/3DPotatoTwin/ground_truth.csv'
    intrinsics = '/mnt/data/PieterBlok/Potato/Data/3DPotatoTwin/1_rgbd/0_camera_intrinsics/realsense_d405_camera_intrinsic.json'
    
    pin_regions = PinRegions(img_root, coco_file, csv_file, intrinsics)
    # pin_regions.visualize_annotations()
    pin_regions.visualize_annotations(visualize_pcd=True)