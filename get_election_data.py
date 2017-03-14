from __future__ import print_function

import subprocess
import os

POST_2010_CENSUS_DISTRICTS = [
    ('http://data-ltsb.opendata.arcgis.com/datasets/657bd028d3e2408fa2fac1af78b0760c_0.zip',
    'data/districts/state_senate_shapefile.zip'),
    ('http://data-ltsb.opendata.arcgis.com/datasets/a907d137f96b49289d83172db8cf96f0_0.zip',
    'data/districts/state_assembly_shapefile.zip'),
    ('http://data-ltsb.opendata.arcgis.com/datasets/b52f3c49baa840adb070e7c56604de59_0.zip',
    'data/districts/congressional_shapefile.zip')
]

ELECTION_2016 = [
    ('http://elections.wi.gov/sites/default/files/Ward%20by%20Ward%20Recount%20%20Canvass%20Results-%20President.xlsx',
    'data/raw/wi_2016_president.xlsx'),
    ('http://elections.wi.gov/sites/default/files/Ward%20by%20Ward%20Report-Assembly.xlsx',
    'data/raw/wi_2016_assembly.xlsx'),
    ('http://elections.wi.gov/sites/default/files/Ward%20by%20Ward%20Report-State%20Senate.xlsx',
    'data/raw/wi_2016_state_senate.xlsx'),
    ('http://elections.wi.gov/sites/default/files/Ward%20by%20Ward%20Report%20-Congress.xlsx',
    'data/raw/wi_2016_congress.xlsx'),
    ('http://elections.wi.gov/sites/default/files/Ward%20by%20Ward%20Report%20-US%20Senator.xlsx',
    'data/raw/wi_2016_senate.xlsx'),
    ('http://data-ltsb.opendata.arcgis.com/datasets/6497103b939d41268a48905631f84de5_0.zip',
    'data/raw/wi_2016_ward_shapefiles.zip')
]

ELECTIONS_2002_2014 = [
    ('http://legis.wisconsin.gov/ltsb/gisdocs/ElectionData/GIS/Wards_fall_2014.shape.zip',
    'data/raw/wi_2014_all.zip'),
    ('http://legis.wisconsin.gov/ltsb/gisdocs/ElectionData/GIS/Wards_111312_ED_110612.zip',
    'data/raw/wi_2012_all.zip'),
    ('http://legis.wisconsin.gov/ltsb/gisdocs/ElectionData/GIS/WISELR_Wards_WTM8391_041712.zip',
    'data/raw/wi_2010_all.zip'),
    ('http://legis.wisconsin.gov/ltsb/gisdocs/ElectionData/GIS/2008_Election_Data_By_Ward.zip',
    'data/raw/wi_2008_all.zip'),
    ('http://legis.wisconsin.gov/ltsb/gisdocs/ElectionData/GIS/2006_Election_Data_By_Ward.shp.zip',
    'data/raw/wi_2006_all.zip'),
    ('http://legis.wisconsin.gov/ltsb/gisdocs/ElectionData/GIS/wards_2004_ed_shp.zip',
    'data/raw/wi_2004_all.zip'),
    ('http://legis.wisconsin.gov/ltsb/gisdocs/ElectionData/GIS/Ward_2002_Election_Data_Shp.zip',
    'data/raw/wi_2002_all.zip')
]

def download_data(f, url, other_options = []):
    if not os.path.exists(f):
        try:
            subprocess.call(['curl', '-o', f, url] + other_options)
            print('Successfully downloaded ', f)
        except:
            print('Failed to download ', f)
    else:
        print(f, 'already exits.')

if __name__ == "__main__":
    folder_paths = [
        'data/', 'data/raw/', 'data/districts'
    ]

    for _path in folder_paths:
        if not os.path.exists(_path):
            os.mkdir(_path)

    all_data = ELECTION_2016 + ELECTIONS_2002_2014 + POST_2010_CENSUS_DISTRICTS
    for url, f in all_data:
        download_data(f, url)
