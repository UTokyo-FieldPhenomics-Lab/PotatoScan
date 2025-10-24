import Metashape
import sys

def select_all_existing_points(pcd):
    chunk.dense_cloud.selectPointsByColor((155,155,155), tolerance = 255) 

doc = Metashape.app.document

if len(sys.argv) > 1 and sys.argv[1] == 'single':
    chunks = [Metashape.app.document.chunk]
else:
    chunks = Metashape.app.document.chunks

for chunk in chunks:
    pcd = chunk.point_cloud

    pcd.setConfidenceFilter(0,3)
    # select all showing images
    pcd.selectPointsByColor((155,155,155), tolerance = 255) 
    # assign class to high noise
    pcd.assignClassToSelection(Metashape.PointClass.HighNoise)

    pcd.resetFilters()
    # clear selection
    pcd.selectPointsByColor((0,0,0), tolerance = 0) 