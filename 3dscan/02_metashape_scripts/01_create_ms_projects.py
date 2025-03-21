import os
import Metashape

import configs as cfg
import ms_utils as mst

if __name__ == '__main__':

    doc_dict = {}

    for subfolder in os.listdir(cfg.image_folder):
        chunk_id = subfolder
        print(chunk_id)

        # 2R5-10 -> aRb-c
        b, c = chunk_id.split('-')
        a, b = b.split('R')
        if a == '':
            a = 1

        a, b, c = int(a), int(b), int(c)

        # 50 as one group maximum
        group_id = b // 5

        fname = f'{a}R_Group{group_id}'

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

    which_scalebar_file = {}
    with open('scalebarlist.csv', "r") as f:
        lines = f.readlines()
        for l in lines:
            c_id, sb_file, _, _, _ = l.split(",")
            which_scalebar_file[c_id] = sb_file

    for ms_file, chunk_rotate in doc_dict.items():

        doc = mst.open_metashape_project(os.path.join(cfg.working_directory, "projects.psx", ms_file+'.psx'))

        for chunk_id, chunk_value in chunk_rotate.items():

            if not '5R' in chunk_id:
                continue

            chunk = mst.create_one_chunk(doc, chunk_id, chunk_value, cfg.camera_mode, cfg.image_folder, cfg.img_format)
            chunk = mst.add_masks(chunk, os.path.join(cfg.working_directory, "masks"))
            chunk = mst.add_scalebar(chunk, scalebar_dict[which_scalebar_file[chunk_id]])

            doc.save()

            # break

        # break
