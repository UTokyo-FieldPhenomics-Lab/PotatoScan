# problems in 2025 data:
# Has duplicate marker id on supportor
# >>> 38(x) 68(x) 98 
# >>> 21(x) 51(x) 81
# >>> 47(x) 77    107

import Metashape

chunks = Metashape.app.document.chunks

keep_markers = [
    "53", "54", "59", "60",
    # "16", "17", "37", "39", "3", "5", "25", "27",
    # "9",  "11", "22", "23", "31", "33", "43", "45"
]


for chunk in chunks:
    markers = chunk.markers

    # marker_id_list = [marker.label for marker in markers]

    # if '98' in marker_id_list:
    #     print(f"Detect chunk [{chunk.label}] has marker id 98, disable marker 38 and 68")
    #     for marker in markers:
    #         if marker.label in ["38", "68"]:
    #             marker.enabled = False

    # if '81' in marker_id_list:
    #     print(f"Detect chunk [{chunk.label}] has marker id 81, disable marker 21 and 51")
    #     for marker in markers:
    #         if marker.label in ["21", "51"]:
    #             marker.enabled = False

    # if '107' in marker_id_list:
    #     print(f"Detect chunk [{chunk.label}] has marker id 107, disable marker 47")
    #     for marker in markers:
    #         if marker.label in ["47"]:
    #             marker.enabled = False

    remove_marker_list = []
    marker_id_list = [marker.label for marker in markers]

    for marker in markers:

        # rename marker id from "target x" -> "x"
        if "target" in marker.label:
            marker_id = str(int(marker.label[7:]))   # target x

            # remove existing detected (53, 54, 59, 60), & redetect as 'target 53'
            if marker_id in marker_id_list:
                remove_marker_list.append(marker)
            else:
                marker.label = marker_id

        if marker.label not in keep_markers:
            remove_marker_list.append(marker)

    chunk.remove(remove_marker_list)

    chunk.remove(chunk.scalebars)
        