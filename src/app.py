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

ALL_SSKMEANS_GEOJSON = {
    'sskmeans{}'.format(i): './static/geojson/sskmeans{}.json'.format(i)
    for i in range(8)
}


@app.route('/')
def main():
    return render_template(
        'index.html', maplink = GMAPS_LINK,
        kmeans_geojson = ALL_SSKMEANS_GEOJSON['sskmeans0']
    )

@app.route('/embed')
def bokeh_geojson():
    """ Embedding a bokeh plot in flask
    """

    # Load GeoJSON sources into a dictionary
    geo_sources = {}
    for step in ['sskmeans{}'.format(i) for i in range(8)]:
        with open(ALL_SSKMEANS_GEOJSON[step]) as f:
            geo_sources[step] = f.read()

    # Set the inital GeoJSONDataSource object
    geo_source = GeoJSONDataSource(
        geojson = geo_sources['sskmeans0']
    )
    # Create a dictionary of all source objects
    sskmeans_sources = {
        key: GeoJSONDataSource(geojson = geo_sources[key])
        for key in geo_sources
    }

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
        title="Generated Wisconsin Districts",
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

    json_patches = fig.patches(
        xs='xs', ys='ys', line_color='black',
        line_width=1, source = geo_source,
        fill_color = {'field': 'id_color'}
    )

    callback_type = CustomJS(
        args = dict(
            source = geo_source,
            sskmeans0 = sskmeans_sources['sskmeans0'],
            sskmeans1 = sskmeans_sources['sskmeans1'],
            sskmeans2 = sskmeans_sources['sskmeans2'],
            sskmeans3 = sskmeans_sources['sskmeans3'],
            sskmeans4 = sskmeans_sources['sskmeans4'],
            sskmeans5 = sskmeans_sources['sskmeans5'],
            sskmeans6 = sskmeans_sources['sskmeans6'],
            sskmeans7 = sskmeans_sources['sskmeans7']
        ),
        code = """
            var f = cb_obj.value;
            if (f == 'kmeans') {
                source.geojson = kmeans_source.geojson;
            } else if (f == 'sskmeans') {
                source.geojson = sskmeans_source.geojson;
            } else if (f == 'sskmeans0') {
                source.geojson = sskmeans0.geojson
            } else if (f == 'sskmeans1') {
                source.geojson = sskmeans1.geojson
            } else if (f == 'sskmeans2') {
                source.geojson = sskmeans2.geojson
            } else if (f == 'sskmeans3') {
                source.geojson = sskmeans3.geojson
            } else if (f == 'sskmeans4') {
                source.geojson = sskmeans4.geojson
            } else if (f == 'sskmeans5') {
                source.geojson = sskmeans5.geojson
            } else if (f == 'sskmeans6') {
                source.geojson = sskmeans6.geojson
            } else if (f == 'sskmeans7') {
                source.geojson = sskmeans7.geojson
            };
            source.trigger('change');
        """
    )
    type_select = Select(
        title = "District Type",
        options = [
            ('sskmeans0', 'SameSizeKMeans Step 0'),
            ('sskmeans1', 'SameSizeKMeans Step 1'),
            ('sskmeans2', 'SameSizeKMeans Step 2'),
            ('sskmeans3', 'SameSizeKMeans Step 3'),
            ('sskmeans4', 'SameSizeKMeans Step 4'),
            ('sskmeans5', 'SameSizeKMeans Step 5'),
            ('sskmeans6', 'SameSizeKMeans Step 6'),
            ('sskmeans7', 'SameSizeKMeans Step 7')
        ], width = int(750/2),
        callback = callback_type
    )

    callback_info = CustomJS(
        args = dict(renderer = json_patches),
        code = """
            var f = cb_obj.value;
            renderer.glyph.fill_color = {'field': f};
            renderer.trigger('change');
        """
    )
    info_select = Select(
        title = "District information",
        options = [
            ('id_color', 'Districts (Categorical)'),
            ('cmpct_col', 'Compactness'),
            ('pdiff_col', 'Population Variance')
        ], width = int(750/2),
        callback = callback_info
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
        css_resources=css_resources
    )
    return encode_utf8(html)

if __name__ == '__main__':
    # Start Flask app
    app.run(host='0.0.0.0', port=PORT, debug=True)
