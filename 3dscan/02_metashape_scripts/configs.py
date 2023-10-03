import os

import platform

user = platform.node()

if user == "Alienware":
    working_directory =  r"C:\Users\kkoda\UTOKYOFieldPhenomics Dropbox\guo ut_fp\dataNprocess\hwang_Pro\data\2023_hokkaido_potato"
elif user == "crest-nerv":
    working_directory =  r"/home/crest/z/hwang_Pro/data/2023_hokkaido_potato"
else:
    raise FileNotFoundError(f"please add new user [{user}] setting in configs.py")


image_folder = os.path.join(working_directory, "images")
save_mask_folder = os.path.join(working_directory, "masks")


####################
# 02 make projects #
####################
metashape_project_path = os.path.join(working_directory, "projects.psx", "align_test.psx")
img_format = "jpg"   # the format of taken images
mask_format = "png"   # the format of output masks, recommended for png format

# 0: fix camera, flip objects, will only detect the markers in the first camera group
# 1: fix object, move camera, will detect the markers in all camera groups
camera_mode = 1


###################
# 04 add referece #
###################

# the XYZ position of each target
# default: None
# target_xyz_position_file = r"E:\2021_tanashi_foldio360\gcp.csv"   

# if XYZ position hard to provide, then can tell the distances between targets instead
# default: None
scalebar_csv_file = "scalebarlist.csv"