import os
import torch
import numpy as np
import random
from pathlib import Path

from loguru import logger
from datetime import datetime

from PIL import Image
import skimage as ski
import supervision as sv

import matplotlib.pyplot as plt

from gdino import GDINO
from ultralytics.models.sam import SAM2DynamicInteractivePredictor


class ImageSegmenter():

    def __init__(self, detect_object:str, logger):
        self.logger = logger

        self.script_root = Path(__file__).parent.resolve()
        self.logger.info(f"Execute the script at {self.script_root}")

        self.gdino_model = GDINO()
        self.gdino_model.build_model(device="cuda")
        self.logger.info(f"Grounding DINO model loaded")

        self.detect_object = detect_object
        # self.texts_prompt = ". ".join(self.detect_objects)
        self.texts_prompt = detect_object
        self.logger.info(f"Prompt for Grounding DINO object detection: {self.texts_prompt}")

        self.sam2_overrides = dict(
            conf=0.01, task="segment", mode="predict", imgsz=1024, 
            model=self.script_root / "checkpoints" / "sam_models" / "sam2.1_b.pt", 
            save=False)
        self.logger.info(f"SAM2 overrides: {self.sam2_overrides}")
        
        self.sam2_predictor = SAM2DynamicInteractivePredictor(overrides=self.sam2_overrides, max_obj_num=3)
        self.logger.info(f"SAM2 predictor loaded")

    @logger.catch
    def apply(self, 
              image_file:str, 
              hsv_threshold=5, hole_fill_ratio=0.05, noise_remove_ratio=0.001, 
              sam2_track=False, draw_figure=False):
        
        image_pil  = Image.open(image_file)
        image_np   = np.asarray(image_pil)

        if not sam2_track:
            # begining SAM2 tracking

            # obtain the bbox
            gdino_results = self.gdino_model.predict([image_pil], [self.texts_prompt], box_threshold=0.3, text_threshold=0.25)
            merged_bbox = self.merge_boxes_of_one_object(gdino_results[0], self.detect_object)

            if merged_bbox is None:
                # not have merged detections, execute source cv mask
                self.logger.info("   => cv mask on full image")
                cleaned_mask = self.get_cv_mask(image_np, hsv_threshold=hsv_threshold, fill_ratio=hole_fill_ratio, remove_ratio=noise_remove_ratio)

                results = self.sam2_predictor(
                    source=image_file,
                    masks=[cleaned_mask], 
                    obj_ids=[1], 
                    update_memory=True)
            else:
                # have merged detections, execute the cv mask only in detection area
                image_np_in_bbox, np_bbox = self.crop_image_from_bbox(image_np, merged_bbox)
                x_min, y_min, x_max, y_max = np_bbox
                self.logger.info(f"   => cv mask on bbox {np_bbox}")

                cleaned_mask_in_bbox = self.get_cv_mask(image_np_in_bbox, hsv_threshold=hsv_threshold, fill_ratio=hole_fill_ratio, remove_ratio=noise_remove_ratio)

                cleaned_mask = np.zeros( (image_np.shape[0], image_np.shape[1]), dtype=bool)
                cleaned_mask[y_min:y_max, x_min:x_max] = cleaned_mask_in_bbox

                results = self.sam2_predictor(
                    source=image_file,
                    bboxes=[merged_bbox.cpu().numpy()], 
                    masks=[cleaned_mask], 
                    obj_ids=[1], 
                    update_memory=True)
            
            if draw_figure:
                gdino_detect_prev = self.draw_gdino_results(image_pil, gdino_results[0])
                cv_mask_prev = self.draw_cv_mask_results(image_pil, cleaned_mask)
                sam2_mask_prev = results[0].plot()

                views = dict(img_np=image_np, gdino=gdino_detect_prev, cv=cv_mask_prev, sam2=sam2_mask_prev)
            else:
                views = {}
            
            return results[0], views
            
        else:
            results = self.sam2_predictor(source=image_file)

            if draw_figure:
                sam2_mask_prev = results[0].plot()
                views = dict(img_np=image_np, gdino=None, cv=None, sam2=sam2_mask_prev)
            else:
                views = {}

            return results[0], views
        
    def refresh_sam2_predictor(self):
        self.sam2_predictor.memory_bank.clear()
        self.sam2_predictor.obj_idx_set.clear()
        self.logger.info("   -> clear SAM2 memory")


    def merge_boxes_of_one_object(self, gdino_result, object_name):
        # Extract the tensor containing all boxes
        all_boxes = gdino_result['boxes']
        labels_array = np.array(gdino_result['text_labels'])

        index_torch = torch.from_numpy(labels_array == object_name)

        # Check if there are any boxes to merge
        if index_torch.sum() > 0:
            # Find the min of all x1 and y1 coordinates
            min_xy = all_boxes[index_torch, :2].min(dim=0).values
            
            # Find the max of all x2 and y2 coordinates
            max_xy = all_boxes[index_torch, 2:].max(dim=0).values
            
            # Concatenate to form the merged bounding box
            merged_box = torch.cat([min_xy, max_xy])
            
            self.logger.info(f"   Merged BBox for object [{object_name}]", merged_box)

            return merged_box
        else:
            self.logger.warning(f"   No bounding boxes to merge for object [{object_name}]")
            self.logger.debug(f"   The gdino_result data is: {gdino_result}")


            return None
        
    @staticmethod
    def get_cv_mask(img_np, hsv_threshold=15, fill_ratio=0.001, remove_ratio=0.001):
        h,w,d = img_np.shape

        # convert to LAB color space
        lab_image = ski.color.rgb2lab(img_np)

        # get color channel
        a_channel = lab_image[:, :, 1]
        b_channel = lab_image[:, :, 2]

        # set channel threshold
        # color_threshold = 15  # 颜色阈值

        # 创建掩膜
        mask = np.logical_or(a_channel > hsv_threshold, b_channel > hsv_threshold)

        # fill holes in the mask
        filled_mask = ski.morphology.remove_small_holes(mask, area_threshold=h*w*fill_ratio)
        cleaned_mask = ski.morphology.remove_small_objects(filled_mask, min_size=h*w*remove_ratio)

        # 将掩膜应用于原始图像
        # result = np.copy(img_np)
        # result[~cleaned_mask] = 0

        return cleaned_mask
    
    @staticmethod
    def crop_image_from_bbox(image_arr: np.ndarray, bbox: list[int]) -> np.ndarray:
        """
        Extracts a region from a NumPy array based on a bounding box.

        The bounding box is expected in the "DL xyxy" format, which is
        [x_min, y_min, x_max, y_max].

        Args:
            image_arr: The source image as a NumPy array (H, W, C) or (H, W).
            bbox: A list or tuple of integers representing the bounding box
                coordinates [x_min, y_min, x_max, y_max].

        Returns:
            A new NumPy array representing the cropped region of the image.
        """
        # Unpack the bounding box coordinates
        x_min, y_min, x_max, y_max = bbox

        # Ensure coordinates are integers for slicing
        x_min, y_min = int(x_min), int(y_min)
        x_max, y_max = int(x_max), int(y_max)

        # Perform the slicing. NumPy's slicing is [y_start:y_end, x_start:x_end]
        # Note that the 'end' index is exclusive in Python slicing.
        cropped_arr = image_arr[y_min:y_max, x_min:x_max]
        
        return cropped_arr, [x_min, y_min, x_max, y_max]


    @staticmethod
    def draw_gdino_results(image, gdino_result, show=False):
        # 2. Extract and convert data
        # The tensors are on the GPU, so we need to move them to the CPU and convert to NumPy
        boxes = gdino_result['boxes'].cpu().numpy()
        scores = gdino_result['scores'].cpu().numpy()
        text_labels = gdino_result['text_labels']

        # Create class_id for each unique label
        unique_labels = list(set(text_labels))
        class_id_map = {label: idx for idx, label in enumerate(unique_labels)}
        class_id = [class_id_map[label] for label in text_labels]

        # 3. Create a supervision.Detections object
        detections = sv.Detections(
            xyxy=boxes,
            confidence=scores,
            class_id=np.array(class_id)
        )

        # 5. Create Annotators
        box_annotator = sv.BoxAnnotator(thickness=4)
        label_annotator = sv.LabelAnnotator(text_scale=2.5, text_thickness=2)

        # 6. Create custom labels for visualization
        # We'll combine the text label with the confidence score
        labels = [
            f"{label} {confidence:.2f}"
            for label, confidence
            in zip(text_labels, scores)
        ]

        # 7. Annotate the image
        annotated_image = box_annotator.annotate(
            scene=image.copy(),
            detections=detections
        )
        annotated_image = label_annotator.annotate(
            scene=annotated_image,
            detections=detections,
            labels=labels
        )

        # 8. Display or save the annotated image
        if show:
            sv.plot_image(annotated_image, size=(12, 12))
        
        return annotated_image
    

    @staticmethod
    def draw_cv_mask_results(image_np, mask, show=False):
        mask_reshaped = mask[np.newaxis, ...]

        boxes = sv.mask_to_xyxy(masks=mask_reshaped)

        # 创建 Detections 对象
        detections = sv.Detections(
            xyxy=boxes,
            mask=mask_reshaped
        )

        # 创建一个 MaskAnnotator 实例
        # 你可以自定义颜色、透明度等
        mask_annotator = sv.MaskAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            opacity=0.8
        )

        # 在原始图像的副本上进行绘制
        annotated_image = mask_annotator.annotate(
            scene=image_np.copy(), 
            detections=detections
        )

        # --- 4. 显示结果 ---

        # supervision 提供了方便的绘图函数
        if show:
            sv.plot_image(annotated_image, size=(10, 10))

        return annotated_image
    
def draw_final_preview(title, save_path, view_dict):
    h, w, _ = view_dict['img_np'].shape
    # Set a base size for the width and calculate height proportionally
    base_size = 10  # inches
    fig_w = base_size
    fig_h = base_size * (h / w)

    if view_dict['gdino'] is not None:
        # draw 2x2 figure
        figsize = (fig_w, fig_h)
        _draw_all_preview(title, save_path, figsize, raw_img=view_dict['img_np'], 
                                            gdino_prev=view_dict['gdino'], 
                                            cv_prev=view_dict['cv'], 
                                            sam_prev=view_dict['sam2'])

    else:
        # draw 1x2 figure
        figsize = (fig_w, fig_h / 2)
        _draw_sam2_inferenced_preview(title, save_path, figsize, raw_img=view_dict['img_np'], 
                                                        sam_prev=view_dict['sam2'])

def _draw_all_preview(title, save_path, fig_size, raw_img, gdino_prev, cv_prev, sam_prev):

    fig, ax = plt.subplots(2, 2, figsize=fig_size)

    plt.suptitle(title)

    ax[0, 0].imshow(raw_img)
    ax[0, 0].axis('off')
    ax[0, 0].set_title("(a) Raw image")

    ax[0, 1].imshow(gdino_prev)
    ax[0, 1].axis('off')
    ax[0, 1].set_title("(b) GDINO detected bbox")

    ax[1, 0].imshow(cv_prev)
    ax[1, 0].axis('off')
    ax[1, 0].set_title("(c) HSV threshed mask")

    ax[1, 1].imshow(sam_prev)
    ax[1, 1].axis('off')
    ax[1, 1].set_title("(d) SAM2 prompted mask")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')

    plt.clf()
    plt.cla()
    plt.close(fig)

    del fig, ax

def _draw_sam2_inferenced_preview(title, save_path, fig_size, raw_img, sam_prev):
    fig, ax = plt.subplots(1, 2, figsize=fig_size)

    plt.suptitle(title)

    ax[0].imshow(raw_img)
    ax[0].axis('off')
    ax[0].set_title("(a) Raw image")

    ax[1].imshow(sam_prev)
    ax[1].axis('off')
    ax[1].set_title("(b) SAM2 tracked mask")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')

    plt.clf()
    plt.cla()
    plt.close(fig)

    del fig, ax

if __name__ == "__main__":
    working_directory = r"/home/crest/w/hwang_Pro/data/202509_sarabetsu_potato/01_sfm_model"

    img_folder = os.path.join(working_directory, 'images')
    mask_folder = os.path.join(working_directory, 'masks')
    preview_directory = os.path.join(mask_folder, 'preview')
    log_folder = os.path.join(working_directory, 'logs')

    logger.add(os.path.join(log_folder, f"gdino_sam2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log") )

    SAM_TRACK_IMG_NUM = 2
    RANDOM_SAVE_POSSIBILITY = 0.1  # save 10% of results for preview

    logger.info(":: init ImageSegmentor")
    imgseg = ImageSegmenter(detect_object="potato tuber", logger=logger)
    logger.info("   ImageSegmentor created")

    if not os.path.exists(preview_directory):
        os.makedirs(preview_directory)
        logger.info(f":: create prview_directory [{preview_directory}]")

    # ========================
    #      start looping
    # ========================
    for foldername, subfolders, filenames in os.walk(img_folder):

        preview_saving_processes = []

        chunk_name = foldername.split('images/')[-1].split('/')[0]

        tracking = 0  # 0 = no track, 1 = track, 2 = track, 3 = track, etc

        if not filenames:
            continue

        logger.info(f"=> precessing [{chunk_name}]")

        imgseg.refresh_sam2_predictor()

        if random.random() > 1 - RANDOM_SAVE_POSSIBILITY:  # save 5% to preview
            save_preview = True
        else:
            save_preview = False

        # ==============================
        #   for each image in subfolder
        # ==============================
        for filename in filenames:
            file_path = os.path.join(foldername, filename)

            mfolder = foldername.replace('images', 'masks')

            if not os.path.exists(mfolder):
                os.makedirs(mfolder)

            maskname = filename.replace('.jpg', '.png').replace('.JPG', '.png')
            mask_path = os.path.join(mfolder, maskname)
            if os.path.exists(mask_path):
                # skip processing exists file
                logger.info(f"   -> image [{file_path}] has been processed, skip")
                continue
            else:
                logger.info(f"   -> Processing image [{file_path}]")

            # mask not exists
            if tracking == 0:
                sam2track = False
            elif tracking > SAM_TRACK_IMG_NUM:
                sam2track = False
                tracking = 0
            else:
                sam2track = True
            
            result, view_dict = imgseg.apply(
                file_path, 5, 0.05, 0.001, sam2_track=sam2track, draw_figure=save_preview
            )

            cv_mask = (result.masks.data.cpu().numpy().squeeze() * 255).astype(np.uint8)

            ski.io.imsave(mask_path, cv_mask)

            if save_preview:
                title = file_path.replace(working_directory, '')

                logger.info("   -> save preview")
                # save preview
                draw_final_preview(
                        title, 
                        os.path.join(preview_directory, f'{chunk_name}_{maskname}'), 
                        view_dict
                )
                logger.info("   <- save preview ends")
            
            tracking += 1