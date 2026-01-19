import pathlib
import sys
import contextlib
import io

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

# convex hull calculation
from scipy.spatial import ConvexHull
import warnings

from loguru import logger


class InsufficientPinPointsError(Exception):
    """
    Raised when pin segmentation finds too few points to compute convex hull.

    This typically occurs when the HSV threshold is too low for the given
    point cloud's color distribution. User should increase the initial
    HSV threshold parameter.

    Parameters
    ----------
    message : str
        Error message describing the issue.
    points_found : int
        Number of points found (less than 4 required for hull).
    threshold : float
        The threshold value that was used.

    Examples
    --------
    >>> raise InsufficientPinPointsError(
    ...     "Too few points for convex hull",
    ...     points_found=2,
    ...     threshold=0.35
    ... )
    """

    def __init__(
        self,
        message: str,
        points_found: int = 0,
        threshold: float = 0.0,
    ) -> None:
        self.points_found = points_found
        self.threshold = threshold
        super().__init__(
            f"{message} (found {points_found} points at threshold {threshold:.2f})"
        )


def _load_coco_silent(coco_file: str) -> COCO:
    """
    Load COCO annotations without printing to stdout.

    The pycocotools library prints progress messages directly to stdout.
    This function captures and redirects those messages to loguru.

    Parameters
    ----------
    coco_file : str
        Path to the COCO JSON annotation file.

    Returns
    -------
    COCO
        The loaded COCO object.

    Examples
    --------
    >>> coco = _load_coco_silent("/path/to/annotations.json")
    """
    # Capture stdout during COCO initialization
    captured_output = io.StringIO()
    with contextlib.redirect_stdout(captured_output):
        coco = COCO(coco_file)

    # Log the captured output at debug level
    output = captured_output.getvalue().strip()
    if output:
        for line in output.split('\n'):
            if line.strip():
                logger.debug(f"COCO: {line.strip()}")

    return coco


class PinRegions:
    """The base object for RgbdPinFetcher
    """
    def __init__(self, img_root, coco_file, csv_file, intrinsics_file):
        self.img_root = img_root
        self.coco = _load_coco_silent(coco_file)
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

    def process_pcd(
        self,
        img,
        dimg,
        bin_potato,
        bin_pin,
        name,
        max_depth_range=150,
        max_depth_contribution=0.02,
        depth_trunc=0.4,
        paint_color=False,
        visualize=False,
    ):
        """
        Process RGBD image to extract pin point cloud.
        
        Parameters
        ----------
        img : np.ndarray
            RGB image.
        dimg : np.ndarray
            Depth image.
        bin_potato : np.ndarray
            Binary mask for potato region.
        bin_pin : np.ndarray
            Binary mask for pin region.
        name : str
            Name for visualization.
        max_depth_range : float, optional
            Max depth range for histogram filtering. Default 150 (2023), use 100 for 2025.
        max_depth_contribution : float, optional
            Max depth contribution for histogram filtering. Default 0.02 (2023), use 0.05 for 2025.
        depth_trunc : float, optional
            Depth truncation for RGBD. Default 0.4 (2023), use 0.5 for 2025.
        paint_color : bool, optional
            Whether to paint pin uniform color.
        visualize : bool, optional
            Whether to show visualization.
            
        Returns
        -------
        o3d.geometry.PointCloud
            Point cloud of the pin region.
        """
        img_potato = np.multiply(img, np.expand_dims(bin_potato, axis=2))
        dimg_potato = np.multiply(dimg, bin_potato)
        dimg_potato_vis = dimg_potato.astype(np.uint8)

        z_min, z_max = self.histogram_filtering(
            dimg, bin_potato, max_depth_range, max_depth_contribution
        )
        dimg_potato[dimg_potato < z_min] = 0
        dimg_potato[dimg_potato > z_max] = 0

        rgb_potato = o3d.geometry.Image((img_potato[:,:,::-1]).astype(np.uint8))
        depth_potato = o3d.geometry.Image(dimg_potato)
        rgbd_potato = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_potato, depth_potato,
            depth_scale=1000.0, depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False
        )
        pcd_potato = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_potato, self.intrinsics
        )

        img_pin = np.multiply(img, np.expand_dims(bin_pin, axis=2))
        dimg_pin = np.multiply(dimg, bin_pin)
        dimg_pin_vis = dimg_pin.astype(np.uint8)

        rgb_pin = o3d.geometry.Image((img_pin[:,:,::-1]).astype(np.uint8))
        depth_pin = o3d.geometry.Image(dimg_pin)
        rgbd_pin = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_pin, depth_pin,
            depth_scale=1000.0, depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False
        )
        pcd_pin = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_pin, self.intrinsics
        )
        if paint_color:
            pcd_pin.paint_uniform_color([1, 1, 0])

        if visualize:
            pcd_potato_vis = pcd_potato.transform(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
            )
            pcd_pin_vis = pcd_pin.transform(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
            )
            o3d.visualization.draw_geometries(
                [pcd_potato_vis, pcd_pin_vis],
                window_name=f"{name} with pin region in yellow"
            )

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
        # Try to load all available coco files and merge them
        coco_file_candidates = [
            rgbd_root / '1_rgbd/pin_regions_2023.json',
            rgbd_root / '1_rgbd/pin_regions_2025.json',
        ]
        
        self.rgbd_root = rgbd_root
        self.img_root = rgbd_root / '1_rgbd/1_image'
        self.csv_file = rgbd_root / '3_pair/ground_truth_2025.csv'
        self.intrinsics_file = rgbd_root / '1_rgbd/0_camera_intrinsics/realsense_d405_camera_intrinsic.json'
        
        # Load all COCO files that exist and create PinRegions for each
        self._coco_objects = {}  # {coco_file_path: COCO object}
        self._pr_objects = {}  # {coco_file_path: PinRegions object}
        all_img_infos_with_source = []  # List of (img_info, coco_file_path)
        
        for coco_file in coco_file_candidates:
            if coco_file.exists():
                coco_file_str = str(coco_file)
                logger.info(f"Loading COCO file: {coco_file}")
                coco = _load_coco_silent(coco_file_str)
                self._coco_objects[coco_file_str] = coco
                
                # Create PinRegions object for this COCO file
                pr = PinRegions(
                    img_root=self.img_root,
                    coco_file=coco_file_str,
                    csv_file=self.csv_file,
                    intrinsics_file=self.intrinsics_file,
                )
                self._pr_objects[coco_file_str] = pr
                
                img_ids = coco.getImgIds()
                img_infos = coco.loadImgs(img_ids)
                # Track source COCO file for each image info
                for info in img_infos:
                    all_img_infos_with_source.append((info, coco_file_str))
        
        if not self._coco_objects:
            raise FileNotFoundError(
                f"No pin_regions JSON found. Tried: {coco_file_candidates}"
            )
        
        # Keep first PinRegions for backward compatibility
        first_coco_file = list(self._pr_objects.keys())[0]
        self.pr = self._pr_objects[first_coco_file]
        
        # Parse all image infos from all COCO files (with source tracking)
        self.img_names = self.parse_img_infos(all_img_infos_with_source)
        logger.debug(f"Loaded {len(self.img_names)} potato IDs from COCO files")
        logger.debug(f"Sample IDs: {list(self.img_names.keys())[:5]}...")

        self.centered_img = self.find_center_img(self.img_names, img_height=720)
        # {'2R1-1': 
        #    {'rgb': '2R1-1/2R1-1_rgb_358.png',
        #     'depth': '2R1-1/2R1-1_depth_358.png',
        #     'coco_id': xxx,
        #     'coco_file': '/path/to/coco.json'},
        #  '2R1-10': 

    def get(self, potato_id, img_id=None, visualize=False, show=False):
        logger.info(f"RgbdPinFetcher.get() called with potato_id={potato_id}")
        
        if img_id is None:
            centered_data = self.centered_img[potato_id]
        else:
            centered_data = self.img_names[potato_id][int(img_id)]
        
        # Get the correct PinRegions object for this potato's COCO file
        coco_file = centered_data.get('coco_file')
        if coco_file and coco_file in self._pr_objects:
            pr = self._pr_objects[coco_file]
            logger.debug(f"Using COCO file: {coco_file} for {potato_id}")
        else:
            pr = self.pr
            logger.warning(f"No COCO file mapping for {potato_id}, using default")
        
        picked_img = {potato_id: centered_data}
        pcd, pin_pcd, pcd_ero, pcd_rela_path = self.get_pcd_pin(pr, picked_img, potato_id)

        pcd_xyz = np.asarray(pcd.points)
        pin_pcd_xyz = np.asarray(pin_pcd.points)

        logger.debug(f"RGBD: pcd has {len(pcd_xyz)} points, pin_pcd has {len(pin_pcd_xyz)} points")

        # get the index of pin
        pin_idx = []
        for p in pin_pcd_xyz:
            distance = pcd_xyz - p
            distance = distance.sum(axis=1)
            idx_temp = np.where(distance == 0)[0]

            if idx_temp:
                pin_idx.append(idx_temp[0])

        logger.debug(f"RGBD: found {len(pin_idx)} pin indices")
        
        if len(pin_idx) == 0:
            logger.warning(f"No pin indices found for {potato_id} - pin segmentation may have failed")

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
    def parse_img_infos(img_infos_with_source):
        """
        Parse image infos from COCO into a structured dict.
        
        Handles two filename formats:
        - 2023: folder/file format like "2R1-1/2R1-1_rgb_100.png"
        - 2025: flat format like "2025-000_rgb_355.png"
        
        Parameters
        ----------
        img_infos_with_source : list
            List of (img_info_dict, coco_file_path) tuples.
        
        Returns
        -------
        dict
            {potato_id: {pos: {'rgb': ..., 'depth': ..., 'coco_id': ..., 'coco_file': ...}}}
        """
        img_names = {}

        for img_info, coco_file in img_infos_with_source:
            fn = img_info['file_name']
            
            # Detect format: check if there's a folder separator
            if '/' in fn:
                # 2023 format: "2R1-1/2R1-1_rgb_100.png"
                img_id = fn.split('/')[0]
                filename = fn.split('/')[1]
            else:
                # 2025 format: "2025-000_rgb_355.png"
                # Extract potato ID: everything before "_rgb_"
                parts = fn.split('_rgb_')
                img_id = parts[0]
                filename = fn

            if img_id not in img_names.keys():
                img_names[img_id] = {}

            # Extract position number from filename
            pos = int(filename.split('_')[-1][:-4])

            if pos not in img_names[img_id].keys():
                img_names[img_id][pos] = {}

            img_names[img_id][pos]['rgb'] = fn
            img_names[img_id][pos]['depth'] = fn.replace('rgb', 'depth')
            img_names[img_id][pos]['coco_id'] = img_info['id']
            img_names[img_id][pos]['coco_file'] = coco_file  # Track source COCO file

        return img_names

    @staticmethod
    def find_center_img(img_names, img_height):
        """
        Pick the most centered image (closest to the half img height).
        
        Parameters
        ----------
        img_names : dict
            {potato_id: {pos: {'rgb': ..., 'depth': ..., 'coco_id': ..., 'coco_file': ...}}}
        img_height : int
            Image height for calculating center position.
            
        Returns
        -------
        dict
            {potato_id: {'rgb': ..., 'depth': ..., 'coco_id': ..., 'coco_file': ...}}
        """
        centered_img = {}

        for potato_id, rgbd_list in img_names.items():
            rgbd_id_array = np.asarray(list(rgbd_list.keys()))
            dis = abs(rgbd_id_array - (img_height / 2))
            min_id = rgbd_id_array[np.argmin(dis)]
            centered_img[potato_id] = rgbd_list[min_id]

        return centered_img

    @staticmethod
    def get_pcd_pin(pr, centered_img, potato_id):
        rgb_rel_path = centered_img[potato_id]['rgb']
        depth_rel_path = centered_img[potato_id]['depth']
        
        # Try to find the image file - handle both folder and flat structures
        # 2023 format: "2R1-1/2R1-1_rgb_100.png" (already has folder)
        # 2025 format in COCO: "2025-000_rgb_355.png" (flat)
        # 2025 actual: "2025-000/2025-000_rgb_355.png" (in folder)
        
        rgb_img_path = pr.img_root / rgb_rel_path
        depth_img_path = pr.img_root / depth_rel_path
        
        # If flat path doesn't exist, try adding potato_id folder prefix
        if not rgb_img_path.exists():
            alt_rgb_path = pr.img_root / potato_id / rgb_rel_path
            alt_depth_path = pr.img_root / potato_id / depth_rel_path
            if alt_rgb_path.exists():
                rgb_img_path = alt_rgb_path
                depth_img_path = alt_depth_path
                logger.debug(f"Using folder-based path for {potato_id}")
            else:
                logger.error(f"Image not found at {rgb_img_path} or {alt_rgb_path}")
        
        # rgb img
        rgba = cv2.imread(str(rgb_img_path), cv2.IMREAD_UNCHANGED)
        if rgba is None:
            raise FileNotFoundError(f"Cannot read RGB image: {rgb_img_path}")
        img = rgba[:,:,:-1]
        mask = rgba[:,:,-1]

        # depth img
        dimg = cv2.imread(str(depth_img_path), cv2.IMREAD_UNCHANGED)

        # annotation on rgb img
        ann_ids = pr.coco.getAnnIds(imgIds=centered_img[potato_id]['coco_id'])
        annotations = pr.coco.loadAnns(ann_ids)

        # Determine year-specific parameters for histogram filtering
        # 2025 format: potato_id starts with "2025-"
        # 2023 format: R1-1, 2R1-1, etc.
        is_2025_data = potato_id.startswith("2025-")
        
        if is_2025_data:
            # 2025 data: use hardcoded values (x3_depth_mm is empty in CSV)
            max_depth_range = 100
            max_depth_contribution = 0.05
            depth_trunc = 0.5
            logger.debug(f"Using 2025 parameters for {potato_id}")
        else:
            # 2023 data: use gt_depth from CSV
            gt_depth_row = pr.df.loc[pr.df['label'] == potato_id, 'x3_depth_mm']
            if gt_depth_row.empty or pd.isna(gt_depth_row.values[0]):
                # Fallback to reasonable defaults
                max_depth_range = 150
                logger.warning(f"No gt_depth found for {potato_id}, using default 150")
            else:
                max_depth_range = gt_depth_row.values[0]
            max_depth_contribution = 0.02
            depth_trunc = 0.4
            logger.debug(f"Using 2023 parameters for {potato_id}, max_depth_range={max_depth_range}")

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
                img, dimg, bin_mask, bin_pin,
                name=ann_ids,
                max_depth_range=max_depth_range,
                max_depth_contribution=max_depth_contribution,
                depth_trunc=depth_trunc,
                paint_color=True,
            )
            
            pcd_ero = pr.process_pcd(
                img, dimg, bin_mask, bin_mask_eros,
                name=ann_ids,
                max_depth_range=max_depth_range,
                max_depth_contribution=max_depth_contribution,
                depth_trunc=depth_trunc,
                paint_color=False,
            )
            
        # read the source image
        rgb_rel_path = centered_img[potato_id]['rgb']
        pcd_filename = rgb_rel_path.replace('rgb', 'pcd').replace('.png', '.ply')
        
        # Handle both folder and flat path formats
        if '/' in rgb_rel_path:
            # 2023 format: "2R1-1/2R1-1_pcd_100.ply"
            pcd_path = pr.img_root / f"../2_pcd" / pcd_filename
        else:
            # 2025 format: need to check if file is in folder
            pcd_path = pr.img_root / f"../2_pcd" / potato_id / pcd_filename
            if not pcd_path.exists():
                # Try flat path
                pcd_path = pr.img_root / f"../2_pcd" / pcd_filename
        
        logger.debug(f"Reading RGBD point cloud: {pcd_path.resolve()}")
        if not pcd_path.exists():
            raise FileNotFoundError(f"RGBD PLY file not found: {pcd_path}")
        pcd = o3d.io.read_point_cloud(str(pcd_path.resolve()))

        return pcd, pcd_pin, pcd_ero, f"1_rgbd/2_pcd/{pcd_filename}"
    
###########
# SfM Pin #
###########
    
class SfMPinFetcher():
    """
    Fetcher for SfM point cloud with pin segmentation.
    
    Pin reference colors are loaded from:
        dataset_root / 2_sfm / 3_pin_refs / {year} / {color}.png
    
    Where year and color are determined from the ground_truth CSV file.
    
    Notes
    -----
    The `status_callback` parameter in `get()` allows passing progress 
    messages to the GUI statusbar during the iterative pin segmentation.
    """

    def __init__(self, dataset_root, csv_file) -> None:
        """
        Initialize SfMPinFetcher.
        
        Parameters
        ----------
        dataset_root : Path
            Root folder of the dataset.
        csv_file : Path
            Path to ground_truth CSV file with measurement_day and pin_color.
        """
        self.dataset_root = pathlib.Path(dataset_root)
        self.sfm_pcd_folder = self.dataset_root / '2_sfm/2_pcd'
        self.pin_ref_folder = self.dataset_root / '2_sfm/3_pin_refs'
        
        # Load ground truth CSV for pin color lookup
        logger.debug(f"Loading CSV: {csv_file}")
        self.df = pd.read_csv(csv_file)
        logger.debug(f"CSV columns: {list(self.df.columns)}")
        logger.debug(f"CSV labels: {self.df['label'].tolist()[:10]}...")
        
        # Cache for loaded reference colors: {(year, color): hsv_array}
        self._ref_color_cache = {}

    def _get_potato_info(self, potato_id):
        """
        Get measurement year and pin color for a potato ID.
        
        Parameters
        ----------
        potato_id : str
            Potato ID (e.g., "2025-000").
            
        Returns
        -------
        tuple[str, str]
            (year, pin_color) e.g., ("2023", "black")
        """
        logger.debug(f"Looking up potato: {potato_id}")
        row = self.df.loc[self.df['label'] == potato_id]
        if row.empty:
            logger.error(f"Potato ID '{potato_id}' not found in CSV")
            logger.debug(f"Available labels: {self.df['label'].tolist()}")
            raise ValueError(f"{potato_id}")
        
        # measurement_day format: "14-09-2023" -> year = "2023"
        measurement_day = row['measurement_day'].values[0]
        year = measurement_day.split('-')[-1]
        
        # pin_color: e.g., "black"
        pin_color = row['pin_color'].values[0]
        
        logger.debug(f"Found: year={year}, pin_color={pin_color}")
        return year, pin_color

    def get_all_pin_colors(self):
        """
        Get a dictionary map of potato ID to pin color.
        
        Returns
        -------
        dict[str, str]
            Map of {potato_id: pin_color}.
        """
        if 'pin_color' not in self.df.columns or 'label' not in self.df.columns:
            return {}
        
        # Filter for rows that have both label and pin_color
        valid_df = self.df.dropna(subset=['label', 'pin_color'])
        return dict(zip(valid_df['label'], valid_df['pin_color']))

    def _get_ref_color_hsv(self, year, color):
        """
        Get reference color HSV for a specific year and color.
        
        Parameters
        ----------
        year : str
            Year folder name (e.g., "2023" or "2025").
        color : str
            Pin color name (e.g., "black", "red").
            
        Returns
        -------
        np.ndarray
            HSV color array [H, S, V].
        """
        cache_key = (year, color)
        if cache_key in self._ref_color_cache:
            return self._ref_color_cache[cache_key]
        
        ref_img_path = self.pin_ref_folder / year / f"{color}.png"
        if not ref_img_path.exists():
            raise FileNotFoundError(
                f"Pin reference image not found: {ref_img_path}"
            )
        
        ref_color_imarray = plt.imread(str(ref_img_path))
        
        # Extract masked pixels (alpha == 1)
        mask = ref_color_imarray[:, :, 3] == 1
        ref_color_masked = ref_color_imarray[mask]
        
        # Convert to HSV and get median
        hsv = np.median(
            skimage.color.rgb2hsv(ref_color_masked[:, 0:3]), 
            axis=0
        )
        
        self._ref_color_cache[cache_key] = hsv
        logger.debug(f"Loaded ref color for {cache_key}: {hsv}")
        return hsv

    def get(
        self,
        potato_id: str,
        thresh: float = None,
        nb_points: int = 40,
        radius: float = 0.005,
        visualize: bool = False,
        show: bool = False,
        status_callback=None,
        hsv_weights: list[float] = None,
        target_hull_volume: float = 100.0,
        threshold_callback=None,
        auto_iteration: bool = True,
    ):
        """
        Get SfM point cloud with pin segmentation.

        Parameters
        ----------
        potato_id : str
            The name of potato, remove the file suffix.
        thresh : float, optional
            The threshold for calculating color differences
            between pin and potato surface, by default None.
        nb_points : int, optional
            Number of points for radius outlier removal.
        radius : float, optional
            Radius for outlier removal.
        visualize : bool
            Return data for visualization.
        show : bool
            Whether show the intermediate results for debugging.
        status_callback : callable, optional
            A callback function with signature `(message: str) -> None`.
            Used to send progress updates to the GUI statusbar.
        hsv_weights : list[float], optional
            HSV channel weights [H, S, V], default [0.8, 0.1, 0.1].
        target_hull_volume : float, optional
            Target hull volume limit in mm³ (default 100.0).
        threshold_callback : callable, optional
            Callback with signature `(threshold: float) -> None`.
            Called during iteration to update UI with current threshold.
        auto_iteration : bool, optional
            If True, iteratively reduce threshold until hull volume is small.
            If False, use only initial threshold (preview mode).

        Returns
        -------
        dict
            Dictionary containing:
            - 'pcd': whole potato point cloud
            - 'pin_pcd': point cloud of pin on potato surface
            - 'pin_idx': the index of pin points in whole potato point cloud

        Raises
        ------
        InsufficientPinPointsError
            When initial threshold yields too few points for convex hull.
        """
        logger.info(f"SfMPinFetcher.get() called with potato_id={potato_id}")

        try:
            # Get year and pin color for this potato
            year, pin_color = self._get_potato_info(potato_id)
            logger.debug(f"Got potato info: year={year}, pin_color={pin_color}")

            # Get reference HSV color
            ref_color_hsv = self._get_ref_color_hsv(year, pin_color)
            logger.debug(f"Got ref color HSV: {ref_color_hsv}")

            # Run HSV-based pin segmentation
            logger.debug(f"Running hsv_ref_pin for {potato_id}")
            rc = self.hsv_ref_pin(
                self.sfm_pcd_folder,
                potato_id,
                ref_color_hsv,
                thresh,
                nb_points,
                radius,
                visualize,
                show,
                status_callback=status_callback,
                hsv_weights=hsv_weights,
                target_hull_volume=target_hull_volume,
                threshold_callback=threshold_callback,
                auto_iteration=auto_iteration,
            )

            rc['pin_pcd'] = rc['pcd'].select_by_index(rc['pin_idx'])
            logger.info(f"Successfully processed {potato_id}")
            return rc

        except InsufficientPinPointsError:
            # Re-raise without wrapping to preserve the specific error
            raise
        except Exception as e:
            logger.exception(f"Error in SfMPinFetcher.get() for {potato_id}")
            raise

    @staticmethod
    def get_hull_volume(o3d_pcd):
        # Check for empty point cloud
        n_points = len(o3d_pcd.points)
        if n_points < 4:
            logger.warning(f"Point cloud has only {n_points} points, cannot compute hull")
            return 0.0
            
        pin_hull = o3d_pcd.compute_convex_hull()[0]
        # still not watertight
        if not pin_hull.is_watertight():
            warnings.warn(
                "Open3d kernel produced a non-watertight convex hull, "
                "using SciPy kernel instead"
            )
            hull = ConvexHull(np.asarray(o3d_pcd.points))
            volume = hull.volume
            return volume * 1000 ** 3  # mm3
        else:
            hull_volume = pin_hull.get_volume() * 1000 ** 3  # mm3
            return hull_volume

    def iter_hull_volume_by_thresh(self, sfm_pcd, color_distance_norm, thresh):

        pin_idx = np.where(color_distance_norm < thresh)[0]
        
        if len(pin_idx) == 0:
            logger.warning(f"No points found with thresh={thresh}")
            return 0.0, pin_idx

        # calculate volume, if too large needs denoise
        pin_pcd = sfm_pcd.select_by_index(pin_idx)
        hull_volume = self.get_hull_volume(pin_pcd)  # mm3

        return hull_volume, pin_idx

    def hsv_ref_pin(
        self,
        sfm_pcd_folder,
        potato_id: str,
        ref_color_hsv,
        thresh: float = None,
        nb_points: int = 40,
        radius: float = 0.005,
        visualize: bool = False,
        show: bool = False,
        status_callback=None,
        hsv_weights: list[float] = None,
        target_hull_volume: float = 100.0,
        threshold_callback=None,
        auto_iteration: bool = True,
    ):
        """
        Perform HSV-based pin segmentation on SfM point cloud.

        Parameters
        ----------
        sfm_pcd_folder : Path
            Path to SfM point cloud folder.
        potato_id : str
            Potato ID.
        ref_color_hsv : np.ndarray
            Reference pin color in HSV format.
        thresh : float, optional
            Color distance threshold. If None, will be auto-determined.
        nb_points : int, optional
            Points for radius outlier removal.
        radius : float, optional
            Radius for outlier removal.
        visualize : bool, optional
            Prepare visualization geometries.
        show : bool, optional
            Show intermediate results.
        status_callback : callable, optional
            Callback for status updates with signature `(message: str) -> None`.
        hsv_weights : list[float], optional
            HSV channel weights [H, S, V], default [0.8, 0.1, 0.1].
        target_hull_volume : float, optional
            Target hull volume limit in mm³ (default 100.0).
        threshold_callback : callable, optional
            Callback with signature `(threshold: float) -> None`.
            Called during iteration to update UI with current threshold.

        Returns
        -------
        dict
            Result container with pin indices and point cloud data.

        Raises
        ------
        InsufficientPinPointsError
            When threshold yields too few points for convex hull.
        """
        # Use default HSV weights if not provided
        if hsv_weights is None:
            hsv_weights = [0.8, 0.1, 0.1]
        HSV_WEIGHT = hsv_weights

        # get the sfm pcd
        sfm_pcd_path = sfm_pcd_folder / potato_id / f"{potato_id}_30000.ply"
        logger.debug(f"Reading SfM point cloud: {sfm_pcd_path}")
        if not sfm_pcd_path.exists():
            raise FileNotFoundError(f"SfM PLY file not found: {sfm_pcd_path}")
        sfm_pcd = o3d.io.read_point_cloud(str(sfm_pcd_path))

        colors = np.asarray(sfm_pcd.colors)
        colors_hsv = skimage.color.rgb2hsv(colors)

        # Calculate color distance using the single ref_color_hsv
        color_distance_diff = abs(colors_hsv - ref_color_hsv)
        # hue -> circular distances
        need_hue_reverse = color_distance_diff[:, 0] > 0.5
        color_distance_diff[need_hue_reverse, 0] = (
            1 - color_distance_diff[need_hue_reverse, 0]
        )

        color_distance_weight = color_distance_diff * np.array(HSV_WEIGHT)
        color_distance = color_distance_weight.sum(axis=1)

        # Normalize color distance to [0, 1] range
        norm = mcolors.Normalize(
            vmin=np.min(color_distance), vmax=np.max(color_distance)
        )
        color_distance_norm = norm(color_distance)

        # Helper to send status updates
        def _update_status(msg: str) -> None:
            logger.info(msg)
            if status_callback is not None:
                status_callback(msg)

        # Helper to update threshold in UI
        def _update_threshold(current_thresh: float) -> None:
            if threshold_callback is not None:
                threshold_callback(current_thresh)

        _update_status("Iterative pin segmentation of SfM point clouds...")

        # Use initial threshold (passed or default 0.35)
        if thresh is None:
            thresh = 0.35

        # Initial segmentation with starting threshold
        hull_volume, pin_idx = self.iter_hull_volume_by_thresh(
            sfm_pcd, color_distance_norm, thresh
        )
        _update_threshold(thresh)

        # Check for insufficient points at initial threshold
        if len(pin_idx) < 4:
            if auto_iteration:
                # Only raise error in auto_iteration mode
                raise InsufficientPinPointsError(
                    "Initial pin points too few. Please increase Initial HSV Threshold.",
                    points_found=len(pin_idx),
                    threshold=thresh,
                )
            else:
                # In preview mode, just log and continue with what we have
                logger.warning(
                    f"Only {len(pin_idx)} pin points found at thresh={thresh:.2f}"
                )

        # Iterative denoise loop (only if auto_iteration is enabled)
        if auto_iteration:
            while hull_volume > target_hull_volume:
                pin_pcd = sfm_pcd.select_by_index(pin_idx)
                pin_pcd_num = len(pin_pcd.points)

                if pin_pcd_num > 10000:
                    msg = (
                        f"Thresh={thresh:.2f}: hull={hull_volume:.1f}mm³ "
                        f"({pin_pcd_num} pts) - too large, reducing threshold"
                    )
                    _update_status(msg)
                    keeped, keeped_idx = pin_pcd, pin_idx
                else:
                    msg = (
                        f"Thresh={thresh:.2f}: hull={hull_volume:.1f}mm³ "
                        f"({pin_pcd_num} pts) - denoising..."
                    )
                    _update_status(msg)
                    keeped, keeped_idx = pin_pcd.remove_radius_outlier(
                        nb_points=min(40, int(pin_pcd_num / 20)), radius=0.005
                    )

                denoised_volume = self.get_hull_volume(keeped)

                if denoised_volume > target_hull_volume:
                    thresh -= 0.05
                    _update_threshold(thresh)

                    if thresh < 0:
                        raise InsufficientPinPointsError(
                            "Threshold reduced below 0. "
                            "Please increase Initial HSV Threshold.",
                            points_found=len(pin_idx),
                            threshold=thresh,
                        )

                    hull_volume, pin_idx = self.iter_hull_volume_by_thresh(
                        sfm_pcd, color_distance_norm, thresh
                    )

                    # Check for insufficient points after reducing threshold
                    if len(pin_idx) < 4:
                        raise InsufficientPinPointsError(
                            "Pin points too few after threshold reduction. "
                            "Please increase Initial HSV Threshold.",
                            points_found=len(pin_idx),
                            threshold=thresh,
                        )
                else:
                    hull_volume = denoised_volume
                    pin_idx = pin_idx[keeped_idx]
                    msg = (
                        f"Pin segmentation complete: thresh={thresh:.2f}, "
                        f"hull={hull_volume:.2f}mm³ (denoised)"
                    )
                    _update_status(msg)
                    _update_threshold(thresh)
                    break
            else:
                # Loop finished without break (hull was already small enough)
                msg = (
                    f"Pin segmentation complete: thresh={thresh:.2f}, "
                    f"hull={hull_volume:.2f}mm³"
                )
                _update_status(msg)
                _update_threshold(thresh)
        else:
            # Preview mode - no iteration, just use initial threshold
            msg = (
                f"Preview mode: thresh={thresh:.2f}, "
                f"hull={hull_volume:.2f}mm³ ({len(pin_idx)} pts)"
            )
            _update_status(msg)
            _update_threshold(thresh)

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