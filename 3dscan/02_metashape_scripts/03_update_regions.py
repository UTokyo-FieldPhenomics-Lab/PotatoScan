import Metashape
import sys

import configs as cfg
from ms_utils import place_region
    
if __name__ == '__main__':
    l = 0.3
    h = 0.02

    for i in range (1, len(sys.argv)):
        arg = sys.argv[i]
        print("Argument " + str(i) + ": " + arg+ "\n")

    if len(sys.argv) > 1 and sys.argv[1] == 'single':
        chunks = [Metashape.app.document.chunk]
    else:
        chunks = Metashape.app.document.chunks

    for chunk in chunks:

        if len(sys.argv) > 1 and sys.argv[1] == 'ref':
            # add GCP for z axis
            chunk.importReference(cfg.target_xyz_position_file, format=Metashape.ReferenceFormat(3), columns="nxyz", delimiter=",")
            chunk.updateTransform()

        place_region(chunk, (0,0, l/2+h), (0.15,0.15,l), 0)