import os
import Metashape

from configs import *

def read_scalebar_csv(scalebar_csv_file):
    if os.path.isfile(scalebar_csv_file):
        scale_dict = {}
        with open(scalebar_csv_file, "r") as f:
            lines = f.readlines()
            for l in lines:
                st, ed, dis = l.split(",")
                scale_dict[st] = {"end":ed, "dis":float(dis)}
    return scale_dict

def open_metashape_project(metashape_project_path):
    if os.path.exists(metashape_project_path):
        doc = Metashape.Document()
        doc.open(metashape_project_path)
    else:
        doc = Metashape.Document()
        doc.save(path=metashape_project_path)

    return doc

def create_one_chunk(doc, chunk_id, chunk_value, camera_mode):
    # update chunk lists
    chunk_name_list = [c.label for c in doc.chunks]

    # examine if already exists:
    if chunk_id in chunk_name_list:
        print(f"[{chunk_id}] already exists")
        return

    print(f"\n\n<========== Generating chunk {chunk_id} =========>\n\n")
    chunk = doc.addChunk()
    chunk.label = chunk_id

    print(f"<--------- Adding images --------->")
    for i, rotate_id in enumerate(chunk_value):
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
        print(f"<--------- Detecting Markers --------->")
        chunk.detectMarkers()

    # rename marker id from "target x" -> "x"
    for marker in chunk.markers:
        if "target" in marker.label:
            marker_id = str(int(marker.label[7:]))   # target x
            marker.label = marker_id

    return chunk

def add_masks(chunk, mask_root_folder):
    print(f"<--------- Adding masks --------->")
    camera_group_list = {}
    for c in chunk.cameras:
        if c.group.label in camera_group_list.keys():
            camera_group_list[c.group.label].append(c)
        else:
            camera_group_list[c.group.label] = [c]
    
    for rotate, camera_list in camera_group_list.items():
        chunk.generateMasks(path=os.path.join(mask_root_folder, chunk.label, rotate, f"{{filename}}.{mask_format}"), masking_mode=Metashape.MaskingModeFile, cameras=camera_list)

    return chunk

def add_scalebar(chunk, scalebar_csv):
    if isinstance(scalebar_csv, dict):
        scale_dict = scalebar_csv
    elif os.path.isfile(scalebar_csv):
        scale_dict = read_scalebar_csv(scalebar_csv)
    else:
        raise TypeError(f'Only file path <str> and outputs <dict> from read_scale_csv() are acceptable, not {type(scalebar_csv)}')

    # update scalebar
    print(f"<--------- Add Scalebar for {chunk.label} --------->")
    # add scalebar
    marker_dict = {marker.label:marker for marker in chunk.markers}

    for marker_id in marker_dict.keys():
        # two scale bar point exists
        if marker_id in scale_dict.keys() and scale_dict[marker_id]["end"] in marker_dict.keys():
            st_id = marker_id
            ed_id = scale_dict[st_id]["end"]
            dis =  scale_dict[st_id]["dis"]

            sc = chunk.addScalebar(marker_dict[st_id], marker_dict[ed_id])
            sc.reference.distance = float(dis)
            sc.reference.accuracy = 0.001

    chunk.updateTransform()

    return chunk

def add_target_position(chunk, target_xyz_position_file):
    # update target position
    if os.path.isfile(target_xyz_position_file):
        print(f"<--------- Add  position for {chunk.label} --------->")
        chunk.importReference(target_xyz_position_file, format=Metashape.ReferenceFormat(3), columns="nxyz", delimiter=",")
        chunk.updateTransform()

    return chunk

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

    scalebar_dict = {}
    for scalebar_csv in ['scalebar.csv', 'scalebar2.csv']:
        scalebar_dict[scalebar_csv] = read_scalebar_csv(scalebar_csv)

    which_scalebar_file = {}
    with open('scalebarlist.csv', "r") as f:
        lines = f.readlines()
        for l in lines:
            c_id, sb_file, _, _, _ = l.split(",")
            which_scalebar_file[c_id] = sb_file

    for ms_file, chunk_rotate in doc_dict.items():

        doc = open_metashape_project(os.path.join(working_directory, "projects.psx", ms_file+'.psx'))

        for chunk_id, chunk_value in chunk_rotate.items():

            if not '5R' in chunk_id:
                continue
            
            chunk = create_one_chunk(doc, chunk_id, chunk_value, camera_mode)
            chunk = add_masks(chunk, os.path.join(working_directory, "masks"))
            chunk = add_scalebar(chunk, scalebar_dict[which_scalebar_file[chunk_id]])

            doc.save()

            # break

        # break
