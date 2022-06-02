from OSMPythonTools.nominatim import Nominatim
from OSMPythonTools.api import Api
nominatim = Nominatim()
api = Api()
def judge(lat, lon):
    if not (-90 <= lat < 90 and -180 <= lat < 180):
        return "input error"
    result = nominatim.query(lat,lon, reverse=True, zoom=17)
    location=result.toJSON()[0]
    osm_type = location["osm_type"]
    osm_id = location["osm_id"]
    query = api.query('{}/{}'.format(osm_type, osm_id))
    tags= query.tags()
    if 'highway' in tags:
      highway=tags['highway']
      print(highway)
      if highway=="motorway":
        road_class = "自専道"
      elif highway=="trunk" or highway=="primary":
        road_class = "幹線道路"
      else:
        road_class = "市街地"
    else:
      road_class = "unclear"
    return road_class
# //test:
# //lat: 纬度  lon:经度
lat=[35.9792043,35.725485, 42.290213]
lon=[119.5107687,119.561771,-83.769023]
for la, lo in zip(lat,lon):
  judge(lat=la, lon=lo)