import os
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from skimage import color, io, morphology

from cascade_psp import Refiner

def get_cv_mask(img_path, color_threshold=15, fill_ratio=0.001, remove_ratio=0.001):
    img_np = plt.imread(img_path)

    h,w,d = img_np.shape

    # convert to LAB color space
    lab_image = color.rgb2lab(img_np)

    # get color channel
    a_channel = lab_image[:, :, 1]
    b_channel = lab_image[:, :, 2]

    # set channel threshold
    # color_threshold = 15  # 颜色阈值

    # 创建掩膜
    mask = np.logical_or(a_channel > color_threshold, b_channel > color_threshold)

    # fill holes in the mask
    filled_mask = morphology.remove_small_holes(mask, area_threshold=h*w*fill_ratio)
    cleaned_mask = morphology.remove_small_objects(filled_mask, min_size=h*w*remove_ratio)


    # 将掩膜应用于原始图像
    result = np.copy(img_np)
    result[~cleaned_mask] = 0

    return cleaned_mask, img_np, result

def save_preview(title, img_np, cv_masks, psp_masks, save_path, random_save=0.05):

    if random.random() > 1-random_save:  # save 5% to preview
        return

    mask_color = np.copy(img_np)
    mask_color[psp_masks==0] = 0

    fig,ax = plt.subplots(2,2, figsize=(10,10))

    plt.suptitle(title)

    ax[0, 0].imshow(img_np)
    ax[0, 0].axis('off')
    ax[0, 0].set_title('raw img')

    ax[0, 1].imshow(cv_masks)
    ax[0, 1].axis('off')
    ax[0, 1].set_title('CV mask')

    ax[1, 0].imshow(psp_masks)
    ax[1, 0].axis('off')
    ax[1, 0].set_title('PSP refined')

    ax[1, 1].imshow(mask_color)
    ax[1, 1].axis('off')
    ax[1, 1].set_title('Refined colored')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')

    plt.clf()
    plt.cla()

    plt.close()

    del fig, ax

if __name__ == '__main__':

    # working_directory = r"/home/crest/z/hwang_Pro/data/2023_hokkaido_potato"
    working_directory = r"/home/crest/w/hwang_Pro/data/2023_tanashi_wheat"
    psp_model_path = 'psp_models/cascade_psp'

    img_folder = os.path.join(working_directory, 'images')
    mask_folder = os.path.join(working_directory, 'masks')
    preview_directory = os.path.join(mask_folder, 'preview')
    psp_refiner = Refiner(device='cuda:0', model_path=psp_model_path)

    if not os.path.exists(preview_directory):
        os.makedirs(preview_directory)

    for foldername, subfolders, filenames in os.walk(img_folder):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            # file_dict[filename] = file_path

            mfolder = foldername.replace('images', 'masks')

            if not os.path.exists(mfolder):
                os.makedirs(mfolder)

            maskname = filename.replace('.jpg', '.png')
            mask_path = os.path.join(mfolder, maskname)
            if os.path.exists(mask_path):
                # skip processing exists file
                continue
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing {file_path.split('images')[-1]}")

            # mask not exists
            cv_mask, img_np, result = get_cv_mask(file_path, 5, 0.00001, 0.001)

            # psp_mask = (cv_mask * 255).astype(np.uint8)
            psp_mask = psp_refiner.refine(img_np, cv_mask*255, fast=False, L=900)

            io.imsave(mask_path, psp_mask, check_contrast=False)  # block low contrast warnings

            title = file_path.replace(working_directory, '')

            if random.random() > 0.95:  # save 5% to preview
                save_preview(title, img_np, cv_mask, psp_mask, os.path.join(preview_directory, maskname))