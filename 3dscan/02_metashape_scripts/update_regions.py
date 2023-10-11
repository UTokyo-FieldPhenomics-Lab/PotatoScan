import Metashape
import math

def placeRegion(chunk, center,size,angle):
    """
    center = (X,Y,Z) # in CRS coordinates
    size = (width,depth,height) # in meters
    angle = float # rotation angle of the region (in degrees)
    """
    crs = chunk.crs
    T = chunk.transform.matrix
    
    center = Metashape.Vector((center[0],center[1],center[2]))
    center = T.inv().mulp(crs.unproject(center))
    
    m = crs.localframe(T.mulp(center)) * T
    # R = m.rotation() * (1. / m.scale())
    R = m.rotation()
    size = Metashape.Vector((size[0],size[1],size[2])) / m.scale()  # scaling the size is required

    print("m.scale: ", m.scale())
    print("size vector:", size)
    
    s = math.sin(angle* math.pi / 180.)
    c = math.cos(angle * math.pi / 180.)

    rot = Metashape.Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    rotation = R.t() * rot
    
    chunk.region.rot = rotation
    chunk.region.center = center
    chunk.region.size = size
    
if __name__ == '__main__':
    l = 0.3
    h = 0.02

    for chunk in Metashape.app.document.chunks:

        # add GCP for z axis
        chunk.importReference('/home/crest/Documents/Github/PotatoScan/3dscan/02_metashape_scripts/gcp.csv', format=Metashape.ReferenceFormat(3), columns="nxyz", delimiter=",")
        chunk.updateTransform()

        placeRegion(chunk, (0,0, l/2+h), (0.15,0.15,l), 0)