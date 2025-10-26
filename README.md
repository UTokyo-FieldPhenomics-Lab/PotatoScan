# 3dscan code (sfm part) of 3DPotatoTwin dataset

## Installation

**Step 1**
Recommend install [uv](https://docs.astral.sh/uv/getting-started/installation/) tool as virtual enviroment manager.

```
$ uv --version
uv 0.6.14
```

Then go to directory `3dscan` and init the virtual env:

```
$ cd /path/to/this/repo/3dscan
$ uv sync
```

It will create a `.venv` at current project folder and install the `tests` dependency group and `train` dependency group inside `pyproject.toml`.

**Step 2**

Download `metashape.whl` which meets your metashape version to `02_metashape_scripts/metasahpe_xxxx.whl`

```
$ uv pip install 02_metashape_scripts/metashape_xxxx.whl
```

## Processing log (2025)

### For preprocessing of masking potato tuber region

> most of the *.ipynb notebook files are draft notes when coding the pipeline, no need to use them

In 2025, instead of using HSV threshold to get rough mask + CascadePSP to refine mask (this is slow, over 4 seconds per frame), we upgraded to:

1. Using grounding DINO (gdino) model to get potato tuber bounding box by text prompt
2. Using previous bbox as prompt for SAM2
3. Using SAM2 video tracking (`SAM2DynamicInteractivePredictor`) to tracking following frames

The prevous workflow works perfect on extracting potato tuber itself. However, the colored pins are often omitted by SAM2.

To solve this problem, we then reuse the HSV threshold like last year to ensure the mask cover colored pins. The workflow is as follows:

1. Using GDINO model to get potato tuber bounding box by text prompt (no change)
2. Apply the HSV threshold just inside the bbox regions (save calcuation time)
3. Using previous bbox **and mask**as prompt for SAM2
4. For the following frames, using SAM2 video tracking
5. For the 4th frame, looping from 1.

Thus, for year 2025, the following files is used:

```
|-- segmask_batch_gdino_sam2.py
|   |--> gdino.py
|   |    |--> auto download weights from internet
|   |--> ultralytics.SAM2
|   |    |--> `checkpoints/sam_models/sam2_b.pt
```

### For 3D reconstruction by Metashape

Need to manually define the `scalebar.csv` and `gcp.csv` files in advance.

* `scalebar.csv` is a table to record `st_marker_id, ed_marker_id, length(m)` used as scalebars to fix model scales.
* `gcp.csv` is a table to record `marker_id, x, y, z` coordinates, at least three points are required to ensure fix the model positions.

In year 2023, we wrongly involved two scalebar systems, thus using `00_reference_list.ipynb` to distinguish them, please ignore in 2025.

The

** 2025, mistakely involved duplicate marker id, thus disable using `/home/crest/w/hwang_Pro/data/202509_sarabetsu_potato/01_sfm_model/scripts/fix_2025_marker_problem.py` to disable duplicated markers by the following rule:

```python
# problems in 2025 data:
# Has duplicate marker id on supportor
# >>> 38(x) 68(x) 98 
# >>> 21(x) 51(x) 81
# >>> 47(x) 77    107

import Metashape

chunks = Metashape.app.document.chunks

for chunk in chunks:
    markers = chunk.markers

    marker_id_list = [marker.label for marker in markers]

    if '98' in marker_id_list:
        print(f"Detect chunk [{chunk.label}] has marker id 98, disable marker 38 and 68")
        for marker in markers:
            if marker.label in ["38", "68"]:
                marker.enabled = False

    if '81' in marker_id_list:
        print(f"Detect chunk [{chunk.label}] has marker id 81, disable marker 21 and 51")
        for marker in markers:
            if marker.label in ["21", "51"]:
                marker.enabled = False

    if '107' in marker_id_list:
        print(f"Detect chunk [{chunk.label}] has marker id 107, disable marker 47")
        for marker in markers:
            if marker.label in ["47"]:
                marker.enabled = False
          
```

## Processing log (2023)

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
