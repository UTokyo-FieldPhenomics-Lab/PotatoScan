import os
import shutil
from zipfile import ZipFile

def zip_subfolders(source_folder, target_folder):
    # Ensure the target folder exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Walk through the source folder
    for root, dirs, files in os.walk(source_folder):
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)
            zip_file_path = os.path.join(target_folder, f"{dir_name}.zip")

            # Create a zip file for each subfolder
            with ZipFile(zip_file_path, 'w') as zip_file:
                for foldername, subdirs, filenames in os.walk(subfolder_path):
                    for filename in filenames:
                        file_path = os.path.join(foldername, filename)
                        # Add file to zip, maintaining folder structure
                        arcname = os.path.relpath(file_path, subfolder_path)
                        zip_file.write(file_path, arcname)
            print(f"Created: {zip_file_path}")

if __name__ == "__main__":
    # source_folder = '/home/crest/w/hwang_Pro/data/2023_hokkaido_potato/images/'
    # target_folder = '/home/crest/Documents/HuggingFace/3DPotatoTwin/2_SfM/0_images/'
    source_folder = '/home/crest/w/hwang_Pro/data/202509_sarabetsu_potato/01_sfm_model/images/'
    target_folder = '/home/crest/w/hwang_Pro/datasets/HuggingFace/3DPotatoTwin/2_sfm/0_images/'
    zip_subfolders(source_folder, target_folder)
