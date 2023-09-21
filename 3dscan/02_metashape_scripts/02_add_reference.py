from configs import *
import Metashape

doc = Metashape.app.document

# read csv file for scalebar is exists
if os.path.isfile(scalebar_csv_file):
    scale_dict = {}
    with open(scalebar_csv_file, "r") as f:
        lines = f.readlines()
        for l in lines:
            st, ed, dis = l.split(",")
            scale_dict[st] = {"end":ed, "dis":float(dis)}
    
for chunk in doc.chunks:
    # update scalebar
    if os.path.isfile(scalebar_csv_file):
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

    # update target position
    if os.path.isfile(target_xyz_position_file):
        print(f"<--------- Add  position for {chunk.label} --------->")
        chunk.importReference(target_xyz_position_file, format=Metashape.ReferenceFormat(3), columns="nxyz", delimiter=",")
        chunk.updateTransform()
    