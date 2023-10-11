import os
import random
import numpy as np
import matplotlib.pyplot as plt

from skimage import color, io, morphology

from cascade_psp import Refiner

from segmask_batch import get_cv_mask, save_preview

if __name__ == '__main__':

    working_directory = r"/home/crest/z/hwang_Pro/data/2023_hokkaido_potato"
    psp_model_path = 'psp_models/cascade_psp'

    img_folder = os.path.join(working_directory, 'images')
    mask_folder = os.path.join(working_directory, 'masks')
    preview_directory = os.path.join(mask_folder, 'preview')
    psp_refiner = Refiner(device='cuda:0', model_path=psp_model_path)

    problem_chunk = ['3R4-4', '3R4-5', '3R4-6', '3R4-7', '3R4-8', '3R4-9', '3R4-10']
    # problem_chunk = ['R1-2']


    if not os.path.exists(preview_directory):
        os.makedirs(preview_directory)

    for foldername, subfolders, filenames in os.walk(img_folder):

        chunk_name = foldername.split('images/')[-1].split('/')[0]

        if chunk_name not in problem_chunk:
            continue
        else:
            print(chunk_name, chunk_name in problem_chunk)

        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            # file_dict[filename] = file_path

            mfolder = foldername.replace('images', 'masks')

            if not os.path.exists(mfolder):
                os.makedirs(mfolder)

            maskname = filename.replace('.jpg', '.png').replace('.JPG', '.png')
            mask_path = os.path.join(mfolder, maskname)
            if os.path.exists(mask_path):
                # skip processing exists file
                # continue
                pass
            else:
                print(f"Processing {file_path}")

            # mask not exists
            cv_mask, img_np, result = get_cv_mask(file_path, 5, 0.05, 0.001)

            psp_mask = psp_refiner.refine(img_np, cv_mask*255, fast=False, L=900)

            io.imsave(mask_path, psp_mask)

            title = file_path.replace(working_directory, '')

            # save 50% to preview
            save_preview(title, img_np, cv_mask, psp_mask, 
                         os.path.join(preview_directory, f'{chunk_name}_{maskname}'),
                         random_save=0.5)