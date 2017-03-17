from __future__ import absolute_import, division, print_function
from builtins import (
    ascii, bytes, chr, dict, filter, hex, input, int, map,
    next, oct, open, pow, range, round, str, super, zip)

import os
import requests

from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__)

PORT = 8090

KMEANS_GEOJSON = (
    'static/geojson/kmeans_districts.json'
)
GMAPS_API_KEY = os.environ['GMAPS_API_KEY']
GMAPS_LINK = (
    "https://maps.googleapis.com/maps/api/" +
    "js?key={}&callback=initMap".format(GMAPS_API_KEY)
)



@app.route('/')
def main():
    print(KMEANS_GEOJSON)
    return render_template(
        'main.html', maplink = GMAPS_LINK,
        kmeans_geojson = KMEANS_GEOJSON
    )


if __name__ == '__main__':
    # Start Flask app
    app.run(host='0.0.0.0', port=PORT, debug=True)
