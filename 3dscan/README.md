# Installation

* Download `metashape.whl` 
* Install `segmentation-refinement`

# Processing log

> most of the *.ipynb notebook files are draft notes when coding the pipeline, no need to use them

* use `00_data_collection` script to collect and organize image data
* use `01_preprocessing/segmask_batch.py` to segment all masks of potatoes
* use `02_metashape_scripts/01_create_ms_projects.py` to create chunks with image and masks added (need a python environemnt with `metashape.whl` installed)
* open Metashape of each project    
    * batch processing `02_metashape_scripts/02_make_mesh.xml`, only checked `align_photos` function
    * execute script `02_metashape_scripts/03_update_regions.py`, to rotate and change the processing regions of model (pass arguments `single` if only need to redo on current chunk rather than all chunks)
    * batch processing `02_metashape_scripts/02_make_mesh.xml`, checked `build_mesh` and `build_texture` function
    * manuall check the results, record the error models
* fix the errors
    * change the color and area threshold in `01_preprocessing/segmask_batch.py` to `01_preprocessing/segmask_batch_fix.py`, re-run
    * create error ones' project by `02_metashape_scripts/04_create_ms_projects_fix.py`
    * repeat the 3D reconstruction steps above
* export models and volumes
    * use `02_metashape_scripts/05_export_models.xml` to export 3D models as obj files
    * use `02_metashape_scripts/06_export_volume.ipynb` to calculate the volumes of each model and save to csv file


