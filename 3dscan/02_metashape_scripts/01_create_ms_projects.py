import os
import Metashape

from loguru import logger
from datetime import datetime

import configs as cfg
import ms_utils as mst

if __name__ == '__main__':

    logger.add(os.path.join(cfg.log_folder, f"create_ms_projects_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"), 
               format="{time} {level} {message}", filter="my_module")

    doc_dict = {}

    metashape_project_name = ""

    for i, subfolder in enumerate( sorted( os.listdir(cfg.image_folder) ) ):

        if subfolder == ".DS_Store":
            continue

        chunk_id = subfolder

        # split to different groups by config.py: project_nax_chunk_num
        if cfg.project_max_chunk_num > 0:
            group_i = i % cfg.project_max_chunk_num
            # update name if go to a new group
            if group_i == 0:
                metashape_project_name = f"{cfg.metashape_project_prefix}_{chunk_id}"

        else:
            metashape_project_name = cfg.metashape_project_prefix

        # create dict key if not exists
        if metashape_project_name not in doc_dict.keys():
            doc_dict[metashape_project_name] = {}

        for subsubfolder in os.listdir(os.path.join(cfg.image_folder, subfolder)):
            rotate_id = subsubfolder
            # check if is an empty folder
            if len(os.listdir(os.path.join(cfg.image_folder, subfolder, subsubfolder))) > 0:
                if chunk_id in doc_dict[metashape_project_name].keys():
                    doc_dict[metashape_project_name][chunk_id].append(rotate_id)
                else:
                    doc_dict[metashape_project_name][chunk_id] = [rotate_id]

    for ms_project_name, chunk_rotate in doc_dict.items():

        doc = mst.open_metashape_project(os.path.join(cfg.working_directory, "projects.psx", ms_project_name+'.psx'))

        for chunk_id, chunk_value in chunk_rotate.items():

            chunk = mst.create_one_chunk(doc, chunk_id, chunk_value, cfg.camera_mode, cfg.image_folder, cfg.img_format)

            if chunk is not None:
                chunk = mst.add_masks(chunk, os.path.join(cfg.working_directory, "masks"), cfg.mask_format)
                if cfg.scalebar_csv_file is not None:
                    chunk = mst.add_scalebar(chunk, cfg.scalebar_csv_file)

                # add GCP for z axis
                if cfg.target_xyz_position_file is not None:
                    chunk.importReference(cfg.target_xyz_position_file, format=Metashape.ReferenceFormat(3), columns="nxyz", delimiter=",")

                chunk.updateTransform()

                doc.save()

            # break

        # break
