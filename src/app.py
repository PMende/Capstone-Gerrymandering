from __future__ import absolute_import, division, print_function
from builtins import (
    ascii, bytes, chr, dict, filter, hex, input, int, map,
    next, oct, open, pow, range, round, str, super, zip)

import os
import requests
import json
from itertools import cycle

from flask import (
    Flask,
    request,
    render_template
)

import numpy as np

from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.models import (
    Range1d,
    PanTool,
    ResetTool,
    WheelZoomTool,
    GeoJSONDataSource,
    LinearColorMapper,
    Select,
    CustomJS
)
from bokeh.layouts import (
    column,
    row
)
import bokeh.palettes
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8

app = Flask(__name__)

PORT = 8090

KMEANS_GEOJSON = (
    './static/geojson/kmeans_districts.json'
)
ALL_GEOJSON = {
    'kmeans': KMEANS_GEOJSON
}
GMAPS_API_KEY = os.environ['GMAPS_API_KEY']
GMAPS_LINK = (
    "https://maps.googleapis.com/maps/api/" +
    "js?key={}&callback=initMap".format(GMAPS_API_KEY)
)



@app.route('/')
def main():
    return render_template(
        'index.html', maplink = GMAPS_LINK,
        kmeans_geojson = KMEANS_GEOJSON
    )

@app.route('/map')
def map():
    return render_template(
        'main.html', maplink = GMAPS_LINK,
        kmeans_geojson = KMEANS_GEOJSON
    )

colors = {
    'Black': '#000000',
    'Red':   '#FF0000',
    'Green': '#00FF00',
    'Blue':  '#0000FF',
}

def getitem(obj, item, default):
    if item not in obj:
        return default
    else:
        return obj[item]

@app.route('/embed')
def polynomial():
    """ Very simple embedding of a polynomial chart
    """

    # Grab the inputs arguments from the URL
    args = request.args

    # Get all the form arguments in the url with defaults
    disttype = 'current'
    distinfo = 'id'


    def_palette = bokeh.palettes.brewer['Set1'][8]
    def_colors = LinearColorMapper(palette=def_palette)
    def_fill_color_dict = {'field': 'id', 'transform': def_colors}

    # Load GeoJSON sources into a dictionary
    geo_sources = {key: None for key in ALL_GEOJSON}
    for key in geo_sources:
        with open(ALL_GEOJSON[key], 'r') as f:
            geo_sources[key] = GeoJSONDataSource(
                geojson = f.read()
            )

    # Set the inital GeoJSON source
    geo_source = geo_sources['kmeans']

    wisc_bounds_long = (-92.8894, -86.764)
    wisc_bounds_lat = (42.4919, 47.0808)

    # Establish the figure bounds
    fig_bounds_buffer = 0.5
    y_factor = 3
    x_bounds = (
        wisc_bounds_long[0] - fig_bounds_buffer,
        wisc_bounds_long[1] + fig_bounds_buffer
    )
    y_bounds = (
        wisc_bounds_lat[0] - fig_bounds_buffer/y_factor,
        wisc_bounds_lat[1] + fig_bounds_buffer/y_factor
    )

    # Find max x interval
    _max_x_interval = (
        wisc_bounds_long[1] - wisc_bounds_long[0]
    ) + fig_bounds_buffer
    _min_x_interval = _max_x_interval/8

    # Define tools to use
    tools = [
        WheelZoomTool(),
        PanTool(),
        ResetTool()
    ]

    fig = figure(
        title="Current and Generated Wisconsin Districts",
        y_range=Range1d(
            bounds = y_bounds,
            start = wisc_bounds_lat[0] - fig_bounds_buffer/y_factor,
            end = wisc_bounds_lat[1] + fig_bounds_buffer/y_factor),
        x_range=Range1d(
            bounds = x_bounds,
            max_interval = _max_x_interval,
            min_interval = _min_x_interval,
            start = wisc_bounds_long[0] - fig_bounds_buffer,
            end = wisc_bounds_long[1] + fig_bounds_buffer),
        tools = tools,
        toolbar_location = 'below',
        active_drag = tools[1],
        active_scroll = tools[0],
        plot_width = 750,
        plot_height = 750
    )

    fig.xaxis.visible = False
    fig.xgrid.visible = False
    fig.yaxis.visible = False
    fig.ygrid.visible = False
    fig.outline_line_width = 3

    colors_dict = {
        'cmpctness': LinearColorMapper(
            palette = bokeh.palettes.brewer['Reds'][9]),
        'id': LinearColorMapper(
            palette = bokeh.palettes.brewer['Set1'][8]),
        'popdiff': LinearColorMapper(
            palette = bokeh.palettes.brewer['RdBu'][11])
    }

    json_patches = fig.patches(
        xs='xs', ys='ys', line_color='black',
        line_width=1, source = geo_sources['kmeans'],
        fill_color = {
            'field': 'id',
            'transform': colors_dict['id']}
    )

    type_select = Select(
        title = "District Type",
        options = [
            ('current', 'Current Districts'),
            ('kmeans', 'Naive KMeans'),
            ('sskmeans', 'SameSizeKMeans')
        ], width = int(750/2),
        # callback = callback_type
    )

    callback_type = CustomJS(
        args = dict(
            renderer = json_patches,
            select = type_select
        ),
        code = """

        """
    )

    # callback_type = CustomJS(
    #     args = dict(
    #         source = geo_source,
    #         current_source = geo_sources['kmeans'],
    #         kmeans_source = geo_sources['kmeans'],
    #         sskmeans_sources = geo_sources['kmeans']
    #     ),
    #     code = """
    #         var geojson = source.geojson;
    #         var f = cb_obj.value;
    #         if (f == 'kmeans') {
    #             geojson = kmeans_source;
    #         } else {
    #             geojson = kmeans_source;
    #         };
    #         source.trigger('change');
    #     """
    # )

    callback_info = CustomJS(
        args = dict(
            source = geo_source,
            current_source = geo_sources['kmeans'],
            kmeans_source = geo_sources['kmeans'],
            sskmeans_sources = geo_sources['kmeans']
        ),
        code = """
            var current_selection =
            var geojson = source.geojson;
            var f = cb_obj.value;
            source.trigger('change');
        """
    )
    info_select = Select(
        title = "District information",
        options = [
            ('id', 'Districts (Categorical)'),
            ('cmpctness', 'Compactness'),
            ('popdiff', 'Population Variance')
        ], width = int(750/2)
    )

    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()

    layout = column(
        row(type_select, info_select),
        fig
    )

    script, div = components(layout)

    html = render_template(
        'embed.html',
        plot_script=script,
        plot_div=div,
        js_resources=js_resources,
        css_resources=css_resources,
        disttype=disttype,
        distinfo=distinfo
    )
    return encode_utf8(html)

if __name__ == '__main__':
    # Start Flask app
    app.run(host='0.0.0.0', port=PORT, debug=True)
