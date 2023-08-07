import pandas as pd
from pyproj import Proj, Geod
import config

p = Proj(proj='utm',zone=31,ellps='WGS84', preserve_units=False)

x,y = p(51.51, 8.95)

lambdafunc = lambda x: pd.Series([
    p(x['longitude'], x['latitude'])[0],
    p(x['longitude'], x['latitude'])[1]
])

lambdafunc2 = lambda x: pd.Series([0, 0]) if x['longitude_2'] == 0 else pd.Series([
    p(x['longitude_2'], x['latitude_2'])[0],
    p(x['longitude_2'], x['latitude_2'])[1]
])

lambdaalvira = lambda x: pd.Series([0, 0]) if x['AlviraTracksTrackPosition_Latitude'] == 0 else pd.Series([
    p(x['AlviraTracksTrackPosition_Longitude'], x['AlviraTracksTrackPosition_Latitude'])[0],
    p(x['AlviraTracksTrackPosition_Longitude'], x['AlviraTracksTrackPosition_Latitude'])[1]
])

lambdaarcus = lambda x: pd.Series([0, 0]) if x['ArcusTracksTrackPosition_Latitude'] == 0 else pd.Series([
    p(x['ArcusTracksTrackPosition_Longitude'], x['ArcusTracksTrackPosition_Latitude'])[0],
    p(x['ArcusTracksTrackPosition_Longitude'], x['ArcusTracksTrackPosition_Latitude'])[1]
])

def testfunc(x):
    #print(f"wanna {x}")
    return 1

p = Proj(proj='utm',zone=31,ellps='WGS84', preserve_units=False)
g = Geod(ellps='WGS84')

csv_file = pd.read_csv('../results/datasetfix2/dataset_BUILD1.csv')
csv_file['drones'] = csv_file.apply(lambda x: "2" if x['scenario_name'] == 'Scenario_2_1' or x['scenario_name'] == 'Scenario_2_2' else "1", axis=1)
csv_file[['utm_x', 'utm_y']] = csv_file.apply(lambdafunc, axis=1)
csv_file[['utm_x2', 'utm_y2']] = csv_file.apply(lambdafunc2, axis=1)

csv_file[['alvira_utm_x', 'alvira_utm_y']] = csv_file.apply(lambdaalvira, axis=1)
csv_file[['arcus_utm_x', 'arcus_utm_y']] = csv_file.apply(lambdaarcus, axis=1)

csv_file['reference_classification'] = csv_file.apply(config.DronesEstimator, args=(True, g, p), axis=1)
csv_file['reference_drones'] = csv_file.apply(lambda x: "0" if x["reference_classification"] == 0 else ("1" if x["reference_classification"] < 5 else "2"), axis=1)

if config.POP_GEOSPATIAL_COORDINATES:
    csv_file.pop("latitude")
    csv_file.pop("longitude")
    csv_file.pop("latitude_2")
    csv_file.pop("longitude_2")
    csv_file.pop("AlviraTracksTrackPosition_Latitude")
    csv_file.pop("AlviraTracksTrackPosition_Longitude")
    csv_file.pop("ArcusTracksTrackPosition_Latitude")
    csv_file.pop("ArcusTracksTrackPosition_Longitude")

csv_file.to_csv("../datasets/merge/dataset_UTM.csv", index=False)

print(csv_file.head())
