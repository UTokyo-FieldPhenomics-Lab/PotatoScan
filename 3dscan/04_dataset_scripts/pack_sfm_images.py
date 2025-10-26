import os
import shutil
from zipfile import ZipFile
from pathlib import Path

def zip_subfolders(source_folder, target_folder):
    source_folder = Path(source_folder)
    target_folder = Path(target_folder)

    # Ensure the target folder exists
    if not target_folder.exists():
        target_folder.mkdir(parents=True)

    # Walk through the source folder
    for subfolder in source_folder.iterdir():
        print(f":: zipping folder {subfolder}")

        if not subfolder.is_dir():
            print(f"=> skip not directroy [{subfolder}]")
            continue

        zip_file_path = target_folder / f"{subfolder.name}.zip"

        # Create a zip file for each subfolder
        with ZipFile(zip_file_path, 'w') as zip_file:
            for subsubfolder in subfolder.iterdir():
                print(f"=> zipping cam folder [{subsubfolder.name}]")

                if not subsubfolder.is_dir():
                    print(f"   -> skip not directory [{subsubfolder}]")
                    continue

                for file in subsubfolder.iterdir():

                    if not file.is_file():
                        print(f"   -> skip not file [{subsubfolder}]")
                        continue

                    arcname = file.relative_to(subfolder)
                    zip_file.write(file, arcname)

        print(f"Created: {zip_file_path}")

if __name__ == "__main__":
    # source_folder = '/home/crest/w/hwang_Pro/data/2023_hokkaido_potato/images/'
    # target_folder = '/home/crest/Documents/HuggingFace/3DPotatoTwin/2_SfM/0_images/'
    source_folder = '/home/crest/w/hwang_Pro/data/202509_sarabetsu_potato/01_sfm_model/images/'
    target_folder = '/home/crest/w/hwang_Pro/datasets/3DPotatoTwin.source/2_sfm/0_images/'
    zip_subfolders(source_folder, target_folder)
