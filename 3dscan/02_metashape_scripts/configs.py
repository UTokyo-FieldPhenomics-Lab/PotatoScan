import os

import platform

user = platform.node()

if user == "Alienware":
    working_directory =  r"C:\Users\kkoda\UTOKYOFieldPhenomics Dropbox\guo ut_fp\dataNprocess\hwang_Pro\data\2023_hokkaido_potato"
elif user == "crest-nerv":
    working_directory =  r"/home/crest/z/hwang_Pro/data/2023_hokkaido_potato"
else:
    raise FileNotFoundError(f"please add new user [{user}] setting in configs.py")

#######################
# 01 make masks by cv #
#######################

image_folder = os.path.join(working_directory, "images")
save_mask_folder = os.path.join(working_directory, "masks")
# 0: not using; -1: using maximum; number: user specific
parallel_computing_core = -1
# use this value to remove noises and filling small holes
small_object_size = 80000 # in our case, = image width * height / 200


#######################
# 01 make masks by dl #
#######################
foreground_labels = "broccoli"

# you can combine several folders together as appended training data
# default: [working_directory]
training_datasets = [working_directory, r"E:\2021_tanashi_foldio360\014_tanashi_broccoli", r"E:\2021_tanashi_foldio360\013_canon_broccoli"]#, r"E:\2021_tanashi_foldio360\012_sweetpotato_batch"]
test_size=0.1  # split to train & valid ratio, here use 10% as valid data
random_state=5907   # random seed number to fix each split the same

# model general
train_background = False
unet_model_name = "unet_model"
psp_model_path = r"E:\hwang_jupyter\08_foldio_segment\unet_pretrain\cascade_psp"

# model configs
# this works on RTX3090 with GPU_RAM=20GB, please downgrade accroding to your GPU performance
batch_size = 64
learning_rate = 1e-3
num_epochs = 100
num_workers = 0  # windows use 0

# model apply
unet_thresh = 0.7   # threshold for UNet Model
psp_thresh = 0.9   # threshold for CascadePSP
overwrite = False   # whether calculate again for existing masks
save_view_ratio = 0.15   # save 15% of preview image

apply_test = False
test_num = 7

# reverse mask to labelme json
mask_noise_threshold = 0.015


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
target_xyz_position_file = r"E:\2021_tanashi_foldio360\gcp.csv"   

# if XYZ position hard to provide, then can tell the distances between targets instead
# default: None
scalebar_csv_file = os.path.join(working_directory, "scripts", "scalebar.csv")


##################
# no need change #
##################

# model directories, no need to change
temp_path = os.path.join(working_directory, "labelme", "models")
if train_background:
    unet_model_path = os.path.join(working_directory, "labelme", "models", unet_model_name + "_bg.pth")
else:
    unet_model_path = os.path.join(working_directory, "labelme", "models", unet_model_name + ".pth")