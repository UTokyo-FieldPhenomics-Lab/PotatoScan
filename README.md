# PotatoScan
Software for yield estimation of potato tubers on a harvester.
<br/><br/>

## Installation
[INSTALL.md](INSTALL.md)
<br/><br/>

## PointRend training
[PointRend_Train.ipynb](PointRend_Train.ipynb)
<br/><br/>

## PointRend inference
[PointRend_Inference.ipynb](PointRend_Inference.ipynb)
<br/><br/>

## Other software scripts
Use a trained PointRend algorithm to auto-annotate unlabelled images: **utils/auto_annotate.py** <br/>

| Argument       	| Description           														|
| ----------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| --img_dir	        | The file directory where the unlabelled images are stored										|
| --img_separator       | String-identifier to separate the relevant frames from the non-relevant frames							|
| --network_config	| Configuration of the backbone of the network												|
| --classes	 	| The names of the classes of the annotated instances											|
| --conf_thres	 	| Confidence threshold of the CNN to do the image analysis										|
| --nms_thres	 	| Non-maximum suppression threshold of the CNN to do the image analysis									|
| --weights_file 	| Weight-file (.pth) of the trained CNN													|
| --export_format	| Specifiy the export-format of the annotations (currently supported formats: **'labelme'**, **'darwin'**)				|
<br/>

**Example syntax (utils/auto_annotate.py):**
```python
python utils/auto_annotate.py --img_dir ./datasets/unlabelled_images --img_separator rgb --network_config ./detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml --classes Potato --conf_thres 0.8 --nms_thres 0.3 --weights_file ./weights/PointRend_Potato/best_model.pth --export_format labelme
```
<br/><br/>

## Contact
This code was developed by Pieter Blok (pieterblok@g.ecc.u-tokyo.ac.jp)
