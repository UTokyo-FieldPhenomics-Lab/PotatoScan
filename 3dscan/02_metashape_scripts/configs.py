import os
import platform

user = platform.node()

if user == "Alienware":
    working_directory =  r"C:\Users\kkoda\UTOKYOFieldPhenomics Dropbox\guo ut_fp\dataNprocess\hwang_Pro\data\2023_hokkaido_potato"
elif user == "crest-nerv":
    working_directory =  r"/home/crest/w/hwang_Pro/data/202509_sarabetsu_potato/01_sfm_model/"
else:
    raise FileNotFoundError(f"please add new user [{user}] setting in configs.py")


image_folder = os.path.join(working_directory, "images")
save_mask_folder = os.path.join(working_directory, "masks")
log_folder = os.path.join(working_directory, "logs")


####################
# 02 make projects #
####################
metashape_project_prefix = "PotatoTuber"
img_format = "jpg"   # the format of taken images
mask_format = "png"   # the format of output masks, recommended for png format

# 0: fix camera, flip objects, will only detect the markers in the first camera group
# 1: fix object, move camera, will detect the markers in all camera groups
camera_mode = 1

# 0 -> put in all chunk
# num -> specific max chunk num
project_max_chunk_num = 50


###################
# 04 add referece #
###################

# the XYZ position of each target
# default: None
target_xyz_position_file = "gcp.csv"  

# if XYZ position hard to provide, then can tell the distances between targets instead
# default: None
scalebar_csv_file = "scalebar.csv"

####################
# update file path #
####################
target_xyz_position_file = os.path.abspath( os.path.join ( os.path.dirname(__file__) , target_xyz_position_file) )
scalebar_csv_file        = os.path.abspath( os.path.join ( os.path.dirname(__file__) , scalebar_csv_file ) )