from __future__ import absolute_import, division, print_function
from builtins import (
    ascii, bytes, chr, dict, filter, hex, input, int, map,
    next, oct, open, pow, range, round, str, super, zip)

import networkx as nx

def shapes_to_graph(shape_list):
    '''Turns a list of shapes into a graph

    Parameters
    ----------
    shape_list: List of dictionaries where dictionaries
        have keys "geoid", and "shape". The value of
        "geoid" is the geoid of the particular shape
        under consideration. The value of "shape" is
        a shapely shape corresponding to that geoid.

    Returns
    ----------
    G : graph whose nodes are geoids. Edges are defined
        between two nodes where the shape of one of the
        nodes touches the shape of the other node
    '''

    G = nx.Graph()

    shape_geoids = [shape['geoid'] for shape in shape_list]
    G.add_nodes_from(shape_geoids)

    touching_shapes = [
        (shape['geoid'], other_shape['geoid'])
        for i, shape in enumerate(shape_list)
        for j, other_shape in enumerate(shape_list)
        if j > i
        if shape['geoid'] != other_shape['geoid']
        if shape['shape'].touches(other_shape['shape'])
    ]
    G.add_edges_from(touching_shapes)

    return G
