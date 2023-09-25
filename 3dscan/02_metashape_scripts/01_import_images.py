import os
import Metashape

from configs import *

if os.path.exists(metashape_project_path):
    doc = Metashape.Document()
    doc.open(metashape_project_path)
else:
    doc = Metashape.Document()
    doc.save(path=metashape_project_path)

chunk_rotates = {}

for subfolder in os.listdir(image_folder):
    chunk_id = subfolder
    print(chunk_id)

    for subsubfolder in os.listdir(os.path.join(image_folder, subfolder)):
        rotate_id = subsubfolder
        # check if is an empty folder
        if len(os.listdir(os.path.join(image_folder, subfolder, subsubfolder))) > 0:
            if chunk_id in chunk_rotates.keys():
                chunk_rotates[chunk_id].append(rotate_id)
            else:
                chunk_rotates[chunk_id] = [rotate_id]

print(chunk_rotates)

# create chunks
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

    print(f"<--------- Adding masks --------->")
    # chunk.generateMasks(path=os.path.join(save_mask_folder, f"{{filename}}.{mask_format}"), masking_mode=Metashape.MaskingModeFile)


doc.save()