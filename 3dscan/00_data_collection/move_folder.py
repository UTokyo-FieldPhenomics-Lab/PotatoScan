import os

image_cache_folder = r"C:\Users\kkoda\OneDrive\画像\digiCam"
image_raw_root = r"C:\Users\kkoda\UTOKYOFieldPhenomics Dropbox\guo ut_fp\dataNprocess\hwang_Pro\data\2023_hokkaido_potato\images"

while True:
    folder_name = input("Please type the sample name\n")

    img_cache_list = os.listdir(image_cache_folder)

    for cimg in img_cache_list:
        
        camera_id = cimg.split('_')[1]

        camera_folder = os.path.join(image_raw_root, folder_name, camera_id)

        if not os.path.exists(camera_folder):
            os.makedirs(camera_folder)

        os.rename(
            os.path.join(image_cache_folder, cimg),
            os.path.join(camera_folder, cimg)
        )

