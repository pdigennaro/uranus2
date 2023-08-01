from pyproj import Proj, Geod
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle as pk

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

UTM = True
RESULTS_BASE_PATH = ""
SAVE_LOG = True
TRAIN_RF = True
DB_SPLIT = True

p = Proj(proj='utm',zone=31,ellps='WGS84', preserve_units=False)
g = Geod(ellps='WGS84')

def saveLog(string):
    if SAVE_LOG:
        f.write(string + "\n")
        f.flush()

# run this only once
if DB_SPLIT:
    dataset_file = '../datasets/merge/dataset_UTM.csv'
    data = pd.read_csv(dataset_file)

    data = data.loc[data["reference_drones"] > 0]

    AlviraTracksTrack_ClassificationENC = LabelEncoder()
    data["AlviraTracksTrack_Classification"] = AlviraTracksTrack_ClassificationENC.fit_transform(data["AlviraTracksTrack_Classification"])

    ArcusTracksTrack_ClassificationENC = LabelEncoder()
    data["ArcusTracksTrack_Classification"] = ArcusTracksTrack_ClassificationENC.fit_transform(data["ArcusTracksTrack_Classification"])
    
    train_set, test_set = train_test_split(data, test_size=0.20, shuffle=True)

    train_set.to_csv("../results/dataset_rf/randomf_train.csv", index=False)
    test_set.to_csv("../results/dataset_rf/randomf_test.csv", index=False)
else:
    # INFO: train and test datasets are connected to cases with references classification > 0 (real drone(s) identified!)
    train_set = pd.read_csv("../results/dataset_rf/randomf_train.csv")
    test_set = pd.read_csv("../results/dataset_rf/randomf_test.csv")

print(f"train: {len(train_set)} --- test: {len(test_set)}")

x = train_set[['AlviraTracksTrackPosition_Altitude',
                    'AlviraTracksTrackVelocity_Azimuth',
                    'AlviraTracksTrackVelocity_Elevation',
                    'AlviraTracksTrackVelocity_Speed',
                    'AlviraTracksTrack_Classification',
                    'AlviraTracksTrack_Score',
                    'ArcusTracksTrackPosition_Altitude',
                    'ArcusTracksTrackVelocity_Azimuth',
                    'ArcusTracksTrackVelocity_Elevation',
                    'ArcusTracksTrackVelocity_Speed',
                    'ArcusTracksTrack_Classification',
                    'ArcusTracksTrack_Score',
                    'reference_drones',
                    'alvira_utm_x',
                    'alvira_utm_y',
                    'arcus_utm_x',
                    'arcus_utm_y']]
y = train_set[['utm_x','utm_y','utm_x2','utm_y2']]
x_test = test_set[['AlviraTracksTrackPosition_Altitude',
                        'AlviraTracksTrackVelocity_Azimuth',
                        'AlviraTracksTrackVelocity_Elevation',
                        'AlviraTracksTrackVelocity_Speed',
                        'AlviraTracksTrack_Classification',
                        'AlviraTracksTrack_Score',
                        'ArcusTracksTrackPosition_Altitude',
                        'ArcusTracksTrackVelocity_Azimuth',
                        'ArcusTracksTrackVelocity_Elevation',
                        'ArcusTracksTrackVelocity_Speed',
                        'ArcusTracksTrack_Classification',
                        'ArcusTracksTrack_Score',
                        'reference_drones',
                        'alvira_utm_x',
                        'alvira_utm_y',
                        'arcus_utm_x',
                        'arcus_utm_y']]
y_test = test_set[['utm_x','utm_y','utm_x2','utm_y2']]

print(x)
print(y)

if TRAIN_RF:
    # create regressor object
    regressor = RandomForestRegressor(n_estimators=250, random_state=0)

    # fit the regressor with x and y data
    regressor.fit(x.values, y)

    print(regressor)
    #print(regressor.oob_score_)

    pk.dump(regressor, open(RESULTS_BASE_PATH + 'regressor.pkl', 'wb'))
else:
    regressor = pk.load(open(RESULTS_BASE_PATH + 'regressor.pkl', 'rb'))
    print(regressor)

print("x0")
print(x.iloc[0, :])

if SAVE_LOG:
    f = open(RESULTS_BASE_PATH + 'regressor.txt', 'a')

medium = 0.0
m_count = 0

medium2 = 0.0
m_count2 = 0

multi_drones = 0

norm_pred = []
norm_truth = []

for ii in range(len(test_set)):
    '''print(f"--- TEST {ii} ---")
    print("ground truth: ", y.iloc[ii, :].values)
    print("prediction: ", regressor.predict([x.iloc[ii, :]]))
    print("---")'''
    prediction = regressor.predict([x_test.iloc[ii, :]])

    print(f"--- TEST {ii} ---\nground truth: {y_test.iloc[ii, :].values}\nprediction: {regressor.predict([x_test.iloc[ii, :]])}\n---")
    saveLog(f"--- TEST {ii} ---\nground truth: {y_test.iloc[ii, :].values}\nprediction: {regressor.predict([x_test.iloc[ii, :]])}\n---")

    ref_drones = int(x_test.iloc[ii, :]["reference_drones"])
    print(f"reference drones: {ref_drones}")

    if ref_drones > 1:
        multi_drones += 1

    dist = 0.0
    dist2 = 0.0

    if UTM:
        (x1, y1) = p(y_test.iloc[ii, :].values[0], y_test.iloc[ii, :].values[1], inverse=True)
        (x3, y3) = p(prediction[0][0], prediction[0][1], inverse=True)
        az11, az12, dist = g.inv(x1, y1, x3, y3)

        ground_truth = y_test.iloc[ii, :].values
        mod_pred = regressor.predict([x_test.iloc[ii, :]])

        if ref_drones > 1:
            (x2, y2) = p(y_test.iloc[ii, :].values[2], y_test.iloc[ii, :].values[3], inverse=True)
            (x4, y4) = p(prediction[0][2], prediction[0][3], inverse=True)
            az22, az23, dist2 = g.inv(x2, y2, x4, y4)

            (x1, y1) = p(ground_truth[0], ground_truth[1], inverse=True)
            (x2, y2) = p(ground_truth[2], ground_truth[3], inverse=True)

            (xx1, yy1) = p(mod_pred[0][0], mod_pred[0][1], inverse=True)
            (xx2, yy2) = p(mod_pred[0][2], mod_pred[0][3], inverse=True)

            norm_truth.append([x1,y1, x2, y2])
            norm_pred.append([xx1, yy1, xx2, yy2])

            medium2 += dist2
            m_count2 += 1
        else:
            (x1, y1) = p(ground_truth[0], ground_truth[1], inverse=True)
            (x2, y2) = p(mod_pred[0][0], mod_pred[0][1], inverse=True)

            norm_truth.append([x1, y1, 0, 0])
            norm_pred.append([x2, y2, 0, 0])

        if dist > 1000:
            print("first DRONE greater...")

        if dist2 > 1000:
            print("second DRONE greater...")

        print(dist, dist2)
        medium += dist
        m_count += 1

mse = mean_squared_error(norm_truth, norm_pred) # Print results
mae = mean_absolute_error(norm_truth, norm_pred)
r2 = r2_score(norm_truth, norm_pred)

if SAVE_LOG:
    f = open(RESULTS_BASE_PATH + 'regressor.txt', 'a')

saveLog(f"\nMSE: {mse}\nMAE: {mae}\nR2: {r2}")
print(f"\nMSE: {mse}\nMAE: {mae}\nR2: {r2}")

print(f"\nMulti drones: {multi_drones}")
print(f"total: {medium / m_count}")
print(f"total2: {medium2 / m_count2}")

if SAVE_LOG:
    f.close()