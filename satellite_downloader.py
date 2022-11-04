# connect to the API
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date
import fiona

api = SentinelAPI('pddk', 'xdv46jyf826401', 'https://apihub.copernicus.eu/apihub')

# search by polygon, time, and SciHub query keywords
footprint = geojson_to_wkt(read_geojson('grid.json'))
products = api.query(footprint,
                     date=('20151219', date(2015, 12, 29)),
                     platformname='Sentinel-2',
                     cloudcoverpercentage=(0, 30))

# download all results from the search
api.download_all(products)

# convert to Pandas DataFrame
products_df = api.to_dataframe(products)

# GeoJSON FeatureCollection containing footprints and metadata of the scenes
api.to_geojson(products)

# GeoPandas GeoDataFrame with the metadata of the scenes and the footprints as geometries
print(api.to_geodataframe(products))
