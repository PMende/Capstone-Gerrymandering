from __future__ import absolute_import, division, print_function
from builtins import (
    ascii, bytes, chr, dict, filter, hex, input, int, map,
    next, oct, open, pow, range, round, str, super, zip)

import os
from itertools import cycle
from functools import partial

import numpy as np

# Imports for working with shapefiles
import pyproj
from shapely.geometry import shape, MultiPolygon, mapping
from shapely.ops import transform, cascaded_union
from descartes import PolygonPatch
import fiona
from fiona.crs import from_epsg

# matplotlib imports
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
from matplotlib.colors import to_rgb, to_hex
from matplotlib import cm
from matplotlib.patches import Polygon

def generate_colors(values, cmap, reference=1):
    _colors = [cmap(value/reference) for value in values]

    return _colors

def plot_shapes(
        shapelist, shape_colors, alpha=0.85, fig_file=None,
        center_of_mass_arr=None, patch_lw = 1.5):
    _patches = [
        PolygonPatch(shape['shape'].intersection(wisconsin))
        for shape in shapelist
    ]

    for patch, color in zip(_patches, cycle(shape_colors)):
        patch.set_facecolor(color)
        patch.set_linewidth(patch_lw)
        patch.set_alpha(alpha)

    fig, ax = plt.subplots()

    fig.patch.set_alpha(0.0)

    for patch in _patches:
        ax.add_patch(patch)

    if center_of_mass_arr is not None:
        ax.plot(center_of_mass_arr[:,0], center_of_mass_arr[:,1])

    ax.relim()
    ax.autoscale_view()
    ax.axis('off')
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    aspect_ratio = (ymax - ymin)/(xmax - xmin)
    x_size = 20
    fig.set_size_inches((x_size, x_size*aspect_ratio))

    if fig_file:
        try:
            fig.savefig(fig_file, bbox_inches='tight')
        except IOError as e:
            raise(e)

    return None

def generate_shapefiles(districts, location,
                        schema, all_schema_values,
                        epsg_spec):

    crs = from_epsg(epsg_spec)
    with fiona.open(location, 'w', 'ESRI Shapefile',
                    schema, crs=crs) as c:
        for district, schema_values in zip(districts, all_schema_values):
            c.write(schema_values)

def geojson_from_shapefile(source, target, simp_factor):

    with fiona.collection(source, "r") as shapefile:
        features = [feature for feature in shapefile]
        crs = " ".join(
            "+{}={}".format(key,value)
            for key, value in shapefile.crs.items()
        )

    for feature in features:
        feature['geometry'] = mapping(
            shape(feature['geometry']).simplify(simp_factor)
        )

    my_layer = {
    "type": "FeatureCollection",
    "features": features,
    "crs": {
        "type": "link",
        "properties": {"href": "kmeans_districts.crs", "type": "proj4"}
        }
    }

    crs_target = os.path.splitext(target)[0]+'.crs'

    with open(target, "w") as f:
        f.write(unicode(json.dumps(my_layer)))
    with open(crs_target, "w") as f:
        f.write(unicode(crs))
