import os
import pandas as pd
import Metashape

from configs import *

def open_metashape_project(metashape_project_path):
    if os.path.exists(metashape_project_path):
        doc = Metashape.Document()
        doc.open(metashape_project_path)
    else:
        doc = Metashape.Document()
        doc.save(path=metashape_project_path)

    return doc

def create_one_chunk(doc, chunk_rotates, mask_root_folder):
    for chunk_id in chunk_rotates.keys():
        # update chunk lists
        chunk_name_list = [c.label for c in doc.chunks]

        # examine if already exists:
        if chunk_id in chunk_name_list:
            print(f"[{chunk_id}] already exists")
            continue


        print(f"\n\n<========== Generating chunk {chunk_id} =========>\n\n")
        chunk = doc.addChunk()
        chunk.label = chunk_id

        print(f"<--------- Adding images --------->")
        for i, rotate_id in enumerate(chunk_rotates[chunk_id]):
            cg = chunk.addCameraGroup()
            cg.label = rotate_id

            img_list = []
            full_path = os.path.join(image_folder, chunk_id, rotate_id)
            all_subfile = os.listdir(full_path)

            print(full_path, all_subfile)
            for s in all_subfile:
                if s.endswith(f".{img_format}"):
                    img_list.append(os.path.join(full_path, s))

            if len(img_list) == 0:
                print(f'can not find any images fits format */.{img_format} from {all_subfile} in {full_path}')
            chunk.addPhotos(img_list, group=i)

            if i == 0 and camera_mode == 0:
                print(f"<--------- Detecting Markers --------->")
                # only detect one rotation markers
                chunk.detectMarkers()

        if camera_mode == 1:
            chunk.detectMarkers()

        # rename marker id from "target x" -> "x"
        for marker in chunk.markers:
            if "target" in marker.label:
                marker_id = str(int(marker.label[7:]))   # target x
                marker.label = marker_id

        doc.save()

        print(f"<--------- Adding masks --------->")
        camera_group_list = {}
        for c in chunk.cameras:
            if c.group.label in camera_group_list.keys():
                camera_group_list[c.group.label].append(c)
            else:
                camera_group_list[c.group.label] = [c]
        
        for rotate, camera_list in camera_group_list.items():
            chunk.generateMasks(path=os.path.join(mask_root_folder, chunk_id, rotate, f"{{filename}}.{mask_format}"), masking_mode=Metashape.MaskingModeFile, cameras=camera_list)

    doc.save()

if __name__ == '__main__':

    doc_dict = {}

    for subfolder in os.listdir(image_folder):
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

        for subsubfolder in os.listdir(os.path.join(image_folder, subfolder)):
            rotate_id = subsubfolder
            # check if is an empty folder
            if len(os.listdir(os.path.join(image_folder, subfolder, subsubfolder))) > 0:
                if chunk_id in doc_dict[fname].keys():
                    doc_dict[fname][chunk_id].append(rotate_id)
                else:
                    doc_dict[fname][chunk_id] = [rotate_id]

    for ms_file, chunk_rotate in doc_dict.items():

        doc = open_metashape_project(os.path.join(working_directory, "projects.psx", ms_file+'.psx'))
        create_one_chunk(doc, chunk_rotate, os.path.join(working_directory, "masks"))