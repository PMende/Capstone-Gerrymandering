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

@app.route('/')
def main():
    return render_template('main.html')


if __name__ == '__main__':
    # Start Flask app
    app.run(host='0.0.0.0', port=PORT, debug=True)
