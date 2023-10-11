import Metashape
import math

chunk = Metashape.app.document.chunk

center_crs = Metashape.Vector((0,0,0.17))

crs = chunk.crs
T = chunk.transform.matrix

center_region = 