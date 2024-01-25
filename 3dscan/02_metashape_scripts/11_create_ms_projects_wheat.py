import os
import Metashape

import configs as cfg
import ms_utils as mst

if __name__ == '__main__':

    doc_dict = {}

    for subfolder in os.listdir(cfg.image_folder):
        chunk_id = subfolder
        print(chunk_id)

        # A005 -> A + 5
        spc = chunk_id[0:1]
        idx = int(chunk_id[1:])

        # 50 as one group maximum
        group_id = (idx - 1) // 50  # start from 1 rather than pythonic 0

        fname = f'{spc}_Group{group_id}'

        if fname not in doc_dict.keys():
            # already exists
            doc_dict[fname] = {}

        for subsubfolder in os.listdir(os.path.join(cfg.image_folder, subfolder)):
            rotate_id = subsubfolder
            # check if is an empty folder
            if len(os.listdir(os.path.join(cfg.image_folder, subfolder, subsubfolder))) > 0:
                if chunk_id in doc_dict[fname].keys():
                    doc_dict[fname][chunk_id].append(rotate_id)
                else:
                    doc_dict[fname][chunk_id] = [rotate_id]

    scalebar_dict = {}
    for scalebar_csv in ['scalebar.csv', 'scalebar2.csv']:
        scalebar_dict[scalebar_csv] = mst.read_scalebar_csv(scalebar_csv)

    for ms_file, chunk_rotate in doc_dict.items():

        doc = mst.open_metashape_project(os.path.join(cfg.working_directory, "projects.psx", ms_file+'.psx'))

        has_chunk_list = [i.label for i in doc.chunks]

        for chunk_id, chunk_value in chunk_rotate.items():

            if chunk_id in has_chunk_list:
                continue

            chunk = mst.create_one_chunk(doc, chunk_id, chunk_value, cfg.camera_mode, cfg.image_folder, cfg.img_format)
            chunk = mst.add_masks(chunk, os.path.join(cfg.working_directory, "masks"), cfg.mask_format)
            chunk = mst.add_scalebar(chunk, 'scalebar2.csv')

            # add GCP for z axis
            chunk.importReference(cfg.target_xyz_position_file, format=Metashape.ReferenceFormat(3), columns="nxyz", delimiter=",")
            chunk.updateTransform()

            doc.save()

            # break

        # break
