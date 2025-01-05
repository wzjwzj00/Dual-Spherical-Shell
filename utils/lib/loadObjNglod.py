
import os
import sys

import numpy as np
import tinyobjloader
import torch

from PIL import Image
def load_obj(fname):


    assert fname is not None and os.path.exists(fname), \
        'Invalid file path and/or format, must be an existing Wavefront .obj'

    reader = tinyobjloader.ObjReader()
    config = tinyobjloader.ObjReaderConfig()
    config.triangulate = True  # Ensure we don't have any polygons

    reader.ParseFromFile(fname, config)

    # Get vertices
    attrib = reader.GetAttrib()
    vertices = torch.FloatTensor(attrib.vertices).reshape(-1, 3)

    # Get triangle face indices
    shapes = reader.GetShapes()
    faces = []
    for shape in shapes:
        faces += [idx.vertex_index for idx in shape.mesh.indices]
    faces = torch.LongTensor(faces).reshape(-1, 3)


    return vertices, faces