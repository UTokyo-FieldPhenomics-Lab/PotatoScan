import pathlib

# RGBD fether
import os
import cv2
import open3d as o3d
from pycocotools.coco import COCO
import numpy as np
import pandas as pd
import copy
import json

from skimage.morphology import erosion, disk

# sfm fetcher
import matplotlib.pyplot as plt
import skimage
import matplotlib.colors as mcolors
from copy import deepcopy


class PinRegions:
    """The base object for RgbdPinFetcher
    """
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


    def process_pcd(self, img, dimg, bin_potato, bin_pin, name, gt_depth, paint_color=False, visualize=False):
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
        if paint_color:
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


class RgbdPinFetcher(object):

    def __init__(self, rgbd_root):
        self.pr = PinRegions(
            img_root = rgbd_root / '1_rgbd/1_image',
            coco_file = rgbd_root / '1_rgbd/pin_regions.json',
            csv_file = rgbd_root / 'ground_truth.csv',
            intrinsics_file = rgbd_root / '1_rgbd/0_camera_intrinsics/realsense_d405_camera_intrinsic.json',
        )

        img_ids = self.pr.coco.getImgIds()
        img_infos = self.pr.coco.loadImgs(img_ids)

        self.centered_img = self.find_center_img(img_infos, img_height=720)

    def get(self, potato_id, visualize=False, show=False):
        pcd, pin_pcd, pcd_ero, pcd_rela_path = self.get_pcd_pin(self.pr, self.centered_img, potato_id)

        pcd_xyz = np.asarray(pcd.points)
        pin_pcd_xyz = np.asarray(pin_pcd.points)

        # get the index of pin
        pin_idx = []
        for p in pin_pcd_xyz:
            distance = pcd_xyz - p
            distance = distance.sum(axis=1)
            idx_temp = np.where(distance == 0)[0]

            if idx_temp:
                pin_idx.append(idx_temp[0])

        if show:
            o3d.visualization.draw_geometries([pcd, pin_pcd])

        results = {
            'pcd': pcd,
            'pin_pcd': pin_pcd,
            'pin_idx': np.asarray(pin_idx),
            'pcd_ero': pcd_ero,
            'pcd_rela_path': pcd_rela_path
        }

        return results


    @staticmethod
    def find_center_img(img_infos, img_height):
        # a dict to store all frame infos
        # it has the following structure
        # {'2R1-1': 
        #    {100: {'rgb'  : '2R1-1/2R1-1_rgb_100.png',
        #           'depth': '2R1-1/2R1-1_depth_100.png'
        #           'coco_id': xxx, },
        #     121: {'rgb'  : '2R1-1/2R1-1_rgb_121.png',
        #           'depth': '2R1-1/2R1-1_depth_121.png',
        #           'coco_id': xxx, },
        #     ...
        img_names = {}

        for img_info in img_infos:
            fn = img_info['file_name']

            img_id = fn.split('/')[0]

            if img_id not in img_names.keys():
                img_names[img_id] = {}

            pos = int(fn.split('_')[-1][:-4])

            if pos not in img_names[img_id].keys():
                img_names[img_id][pos]  = {}

            img_names[img_id][pos]['rgb'] = fn
            img_names[img_id][pos]['depth'] = fn.replace('rgb', 'depth')
            img_names[img_id][pos]['coco_id'] = img_info['id']

        # pick the most centered one (closest to the half img height)
        # it has the following structure:
        # {
        # '2R1-1': 
        #    {'rgb': '2R1-1/2R1-1_rgb_358.png',
        #     'depth': '2R1-1/2R1-1_depth_358.png',
        #     'coco_id': xxx},
        # '2R1-10': 
        #    {'rgb': '2R1-10/2R1-10_rgb_364.png',
        #     'depth': '2R1-10/2R1-10_depth_364.png',
        #     'coco_id': xxx},
        centered_img = {}

        for potato_id, rgbd_list in img_names.items():

            rgbd_id_array = np.asarray(list(rgbd_list.keys())) 

            dis = abs(rgbd_id_array - ( img_height / 2 ))
            
            min_id = rgbd_id_array[np.argmin(dis)]

            centered_img[potato_id] = rgbd_list[min_id]

        return centered_img

    @staticmethod
    def get_pcd_pin(pr, centered_img, potato_id):
        rgb_img_path = pr.img_root / centered_img[potato_id]['rgb']
        depth_img_path = pr.img_root / centered_img[potato_id]['depth']

        # rgb img
        rgba = cv2.imread(str(rgb_img_path), cv2.IMREAD_UNCHANGED)
        img = rgba[:,:,:-1]
        mask = rgba[:,:,-1]

        # depth img
        dimg = cv2.imread(str(depth_img_path), cv2.IMREAD_UNCHANGED)

        # annotation on rgb img
        ann_ids = pr.coco.getAnnIds(imgIds=centered_img[potato_id]['coco_id'])
        annotations = pr.coco.loadAnns(ann_ids)

        # the x3_depth_mm of given potato
        gt_depth = pr.df.loc[pr.df['label'] == potato_id, 'x3_depth_mm'].values[0]

        if len(annotations) != 1:
            raise ValueError('has multiple annotations for one potato')
        else:
            pin_region = annotations[0]['segmentation']
            pin_mask = pr.binary_mask(img, pin_region)
            bin_pin = pin_mask.astype(bool)
            bin_mask = mask.astype(bool)

            # remove the boundary fly overs
            footprint = disk(6)
            bin_mask_eros = erosion(bin_mask, footprint)

            pcd_pin = pr.process_pcd(
                img, dimg, bin_mask, bin_pin, paint_color=True,
                name=ann_ids, gt_depth=gt_depth)
            
            pcd_ero = pr.process_pcd(
                img, dimg, bin_mask, bin_mask_eros, paint_color=False, 
                name=ann_ids, gt_depth=gt_depth)
            
        # read the source image
        pcd_file = centered_img[potato_id]['rgb'].replace('rgb', 'pcd').replace('.png', '.ply')
        pcd_path = pr.img_root / f"../2_pcd" / pcd_file
        pcd = o3d.io.read_point_cloud(str(pcd_path.resolve()))

        return pcd, pcd_pin, pcd_ero, f"1_rgbd/2_pcd/{pcd_file}"
    
###########
# SfM Pin #
###########
    
class SfMPinFetcher():

    def __init__(self, dataset_root, ref_folder) -> None:
        self.sfm_pcd_folder = dataset_root / '2_SfM/2_pcd'

        self.ref_color_hsv = self.get_ref_color(ref_folder)

    def get(self, potato_id, 
            thresh=None, # for color diff
            nb_points=40, radius=0.005, # for denoise
            visualize=False, show=False
        ):
        """All the default settings are according to POTATO!
        Refer to `05_mesh_pin_colorref.ipynb` for draft references

        Parameters
        ----------
        potato_id : str
            The name of potato, remove the file suffix
        thresh : float, optional
            The threshold for calcuating color differences 
            between pin and potato surface, by default None
        nb_points | radius : float, opional
            The number of points to denoise point cloud, 
            the parameter of `remove_radius_outlier` in open3d
        visualize: bool
            Return data for visualization
        show: bool
            Whether show ths intermediate results for debugging

        Returns
        -------
        sfm_pcd
            whole potato point cloud
        sfm_pin_pcd
            point cloud of pin on potato surface 
        sfm_pin_idx
            the index of pin points in whole potato point cloud
        """

        # rc -> results_container
        rc = self.hsv_ref_pin(
            self.sfm_pcd_folder, potato_id, self.ref_color_hsv, 
            thresh, nb_points, radius, visualize, show)
        
        rc['pin_pcd'] = rc['pcd'].select_by_index(rc['pin_idx'])

        return rc

    @staticmethod
    def get_ref_color(ref_folder):

        ref_color_folder = pathlib.Path(ref_folder)

        ref_color_rgb = []
        ref_color_hsv = {}

        for pin_img_file in os.listdir(ref_color_folder):

            pin_id = pin_img_file.replace('.png', '')

            ref_img_path = ref_color_folder / pin_img_file

            ref_color_imarray = plt.imread( str(ref_img_path) )

            mask = ref_color_imarray[:,:,3] == 1

            ref_color_masked = ref_color_imarray[mask]

            ref_color_rgb.append(np.median(ref_color_masked[:,0:3], axis=0))
            ref_color_hsv[pin_id] = np.median(skimage.color.rgb2hsv(ref_color_masked[:,0:3]), axis=0)

        # custom_colormap = ListedColormap(np.asarray(ref_color_rgb))

        return ref_color_hsv

    @staticmethod
    def get_hull_volume(o3d_pcd):
        pin_hull = o3d_pcd.compute_convex_hull()[0]
        hull_volume = pin_hull.get_volume() * 1000 ** 3 # mm3

        return hull_volume

    def iter_hull_volume_by_thresh(self, sfm_pcd, color_distance_norm, thresh):

        pin_idx = np.where(color_distance_norm < thresh)[0]

        # calculate volume, if too large needs denoise
        pin_pcd = sfm_pcd.select_by_index(pin_idx)
        hull_volume = self.get_hull_volume(pin_pcd) # mm3

        return hull_volume, pin_idx

    def hsv_ref_pin(self,
        sfm_pcd_folder, potato_id, 
        ref_color_hsv, thresh=None, # for color diff
        nb_points=40, radius=0.005, # for denoise
        visualize=False, show=False
    ):
        # get the sfm pcd
        sfm_pcd_path = sfm_pcd_folder / potato_id / f"{potato_id}_30000.ply"
        sfm_pcd = o3d.io.read_point_cloud( str(sfm_pcd_path) )

        colors = np.asarray(sfm_pcd.colors)

        colors_hsv = skimage.color.rgb2hsv(colors)

        
        # color_distance = abs(colors_hsv - ref_color_hsv[ potato_id.split('-')[-1] ]).sum(axis=1)
        color_distance_diff = abs(colors_hsv - ref_color_hsv[ potato_id.split('-')[-1] ])
        # hue -> circular distances
        need_hue_reverse = color_distance_diff[:,0] > 0.5
        color_distance_diff[need_hue_reverse, 0] = 1 - color_distance_diff[need_hue_reverse, 0]

        HSV_WEIGHT = [0.8,0.1,0.1]
        color_distance_weight = color_distance_diff * np.array(HSV_WEIGHT)  # hsv weight
        color_distance = color_distance_weight.sum(axis=1)

        # 定义一个Normalize对象，用于将数据值归一化到[0, 1]的范围
        norm = mcolors.Normalize(vmin=np.min(color_distance), vmax=np.max(color_distance))

        color_distance_norm = norm(color_distance)

        print(":: iterative pin segmentation of SfM point clouds")

        # manually set the threshold
        if thresh is not None:
            hull_volume, pin_idx = self.iter_hull_volume_by_thresh(sfm_pcd, color_distance_norm, thresh)
        
        # looping the thresh to denoise
        else:
            thresh = 0.35
            hull_volume, pin_idx = self.iter_hull_volume_by_thresh(sfm_pcd, color_distance_norm, thresh)

            while hull_volume > 60:
                pin_pcd = sfm_pcd.select_by_index(pin_idx)
                pin_pcd_num = len(pin_pcd.points)
                print(f"Thresh={thresh} get pin convex hull volumn {hull_volume} > 60 with [{pin_pcd_num}] points, denoise first")
                keeped, keeped_idx = pin_pcd.remove_radius_outlier(nb_points=min(40, int(pin_pcd_num/20)), radius=0.005)

                denoised_volume = self.get_hull_volume(keeped)

                if denoised_volume > 60:  # still > 50 after denoising
                    thresh -= 0.05

                    if thresh <0:
                        raise ValueError(" x   Threshold can not below 0")

                    hull_volume, pin_idx = self.iter_hull_volume_by_thresh(sfm_pcd, color_distance_norm, thresh)
                else:
                    hull_volume = denoised_volume
                    pin_idx = pin_idx[keeped_idx]
                    print(f"   Stop at thresh={thresh} with hull volume = {hull_volume} after denoising")
                    break
            else:
                print(f"   Stop at thresh={thresh} with hull volume = {hull_volume}")

        results_container = {
            "pin_idx": pin_idx,
            "pcd": sfm_pcd,
            "pcd_rela_path": f"2_sfm/1_mesh/{potato_id}/{potato_id}.obj" ,
            "stop_thresh": thresh,
            "stop_hull_volume": hull_volume,
            "hsv_weight": HSV_WEIGHT,
        }

        if visualize or show:
            # 选择一个colormap
            colormap = plt.cm.viridis

            # 使用colormap和Normalize对象将数据值映射到颜色
            color_array = colormap(norm(color_distance))
            sfm_pcd_cm = deepcopy(sfm_pcd)
            sfm_pcd_cm.colors = o3d.utility.Vector3dVector(color_array[:,0:3])

            # add offsets
            xyz = np.asarray(sfm_pcd_cm.points) + np.array([0.1, 0, 0])
            sfm_pcd_cm.points = o3d.utility.Vector3dVector(xyz)

            pin_pcd = sfm_pcd.select_by_index(pin_idx)
            pin_id = potato_id.split('-')[-1] 
            if pin_id == '3': # red pin
                pin_pcd.paint_uniform_color([1,1,0])
            else:
                pin_pcd.paint_uniform_color([1,0,0])

            results_container['pcd_offset_colormap'] = sfm_pcd_cm
            results_container['pin_pcd_strengthen'] = pin_pcd
            
            if show:
                o3d.visualization.draw_geometries([sfm_pcd, sfm_pcd_cm, pin_pcd], window_name=f"{potato_id} | thresh={thresh}")

        return results_container
    
            
if __name__ == '__main__':
    # example for pin_regions
    img_root = '/mnt/data/PieterBlok/Potato/Data/3DPotatoTwin/1_rgbd/1_image'
    coco_file = '/mnt/data/PieterBlok/Potato/Data/3DPotatoTwin/1_rgbd/pin_regions.json'
    csv_file = '/mnt/data/PieterBlok/Potato/Data/3DPotatoTwin/ground_truth.csv'
    intrinsics = '/mnt/data/PieterBlok/Potato/Data/3DPotatoTwin/1_rgbd/0_camera_intrinsics/realsense_d405_camera_intrinsic.json'
    
    pin_regions = PinRegions(img_root, coco_file, csv_file, intrinsics)
    # pin_regions.visualize_annotations()
    pin_regions.visualize_annotations(visualize_pcd=True)

    # an advanced wrapper
    rgbd_root = pathlib.Path(r'/home/crest/w/hwang_Pro/datasets/3DPotatoTwin')
    rgbd_fetcher = RgbdPinFetcher(rgbd_root)
    pcd, pcd_pin = rgbd_fetcher.get('2R1-1')