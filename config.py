from enum import Enum

import torch
from torch import nn

MIN_DISTANCE_CLASSIFICATION_METERS = 50

CASE_UNKNOWN = 0
CASE_FIXED_WING = 1
CASE_MAVIC_PRO = 2 # aladrian-MAVIC PRO
CASE_PHANTOM_4_PRO = 3 # kcdgc-P4 Professional V2.0
CASE_MAVIC2 = 4 # djiuser_97p9AXasssb6-Mavic2
CASE_PHANTOM4PRO_MAVIC2 = 5
CASE_PHANTOM4PRO_MAVICPRO = 6

POP_GEOSPATIAL_COORDINATES = False
PRINT_LOGS = False

class Net(torch.nn.Module):
    def __init__(self, n_features):
        super(Net, self).__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(n_features, 512 * 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512 * 512, 7)
        )

    def forward(self, x):
        return self.layers(x)

class Classifications(Enum):
    CASE_UNKNOWN = 0
    CASE_FIXED_WING = 1
    CASE_MAVIC_PRO = 2  # aladrian-MAVIC PRO
    CASE_PHANTOM_4_PRO = 3  # kcdgc-P4 Professional V2.0
    CASE_MAVIC2 = 4  # djiuser_97p9AXasssb6-Mavic2
    CASE_PHANTOM4PRO_MAVIC2 = 5
    CASE_PHANTOM4PRO_MAVICPRO = 6

def DronesEstimator(input, print_logs, g, p):
    val_alvira = input['AlviraTracksTrack_Classification']  # AlviraTracksTrack_Classification
    val_arcus = input['ArcusTracksTrack_Classification']  # ArcusTracksTrack_Classification
    val_diana = input['DianaTargetsTargetClassification_type']  # DianaTargetsTargetClassification_type

    result = CASE_UNKNOWN
    scenario = input['scenario_name']

    firstDrone = False
    secondDrone = False

    xx = input["longitude"]
    yy = input["latitude"]
    xx2 = input["longitude_2"]
    yy2 = input["latitude_2"]
    alv_x = input["AlviraTracksTrackPosition_Longitude"]
    alv_y = input["AlviraTracksTrackPosition_Latitude"]
    arc_x = input["ArcusTracksTrackPosition_Longitude"]
    arc_y = input["ArcusTracksTrackPosition_Latitude"]

    if scenario == 'Scenario_Parrot_a':
        az12, az21, dist = g.inv(xx, yy, alv_x, alv_y)
        az12, az21, dist2 = g.inv(xx, yy, arc_x, arc_y)

        if print_logs:
            print(f"({xx}, {yy});\n({arc_x}, {arc_y}); ({alv_x}, {alv_y})")
            print(f"1_alv: {dist}, 1_arc: {dist2}")

        if dist < MIN_DISTANCE_CLASSIFICATION_METERS or dist2 < MIN_DISTANCE_CLASSIFICATION_METERS:
            if print_logs:
                print(f"{bcolors.OKGREEN}Drone FOUND!{bcolors.ENDC}")
            result = CASE_FIXED_WING

    if scenario == "Scenario_1_1":
        az12, az21, dist = g.inv(xx, yy, alv_x, alv_y)
        az12, az21, dist2 = g.inv(xx, yy, arc_x, arc_y)

        if print_logs:
            print(f"({xx}, {yy});\n({arc_x}, {arc_y}); ({alv_x}, {alv_y})")
            print(f"1_alv: {dist}, 1_arc: {dist2}")

        if dist < MIN_DISTANCE_CLASSIFICATION_METERS or dist2 < MIN_DISTANCE_CLASSIFICATION_METERS:
            if print_logs:
                print(f"{bcolors.OKGREEN}Drone FOUND!{bcolors.ENDC}")
            result = CASE_MAVIC_PRO
    if scenario == "Scenario_1_2_b":
        az12, az21, dist = g.inv(xx, yy, alv_x, alv_y)
        az12, az21, dist2 = g.inv(xx, yy, arc_x, arc_y)

        if print_logs:
            print(f"({xx}, {yy});\n({arc_x}, {arc_y}); ({alv_x}, {alv_y})")
            print(f"1_alv: {dist}, 1_arc: {dist2}")

        if dist < MIN_DISTANCE_CLASSIFICATION_METERS or dist2 < MIN_DISTANCE_CLASSIFICATION_METERS:
            if print_logs:
                print(f"{bcolors.OKGREEN}Drone FOUND!{bcolors.ENDC}")
            result = CASE_PHANTOM_4_PRO
    if scenario == "Scenario_1_3":
        az12, az21, dist = g.inv(xx, yy, alv_x, alv_y)
        az12, az21, dist2 = g.inv(xx, yy, arc_x, arc_y)

        if print_logs:
            print(f"({xx}, {yy});\n({arc_x}, {arc_y}); ({alv_x}, {alv_y})")
            print(f"1_alv: {dist}, 1_arc: {dist2}")

        if dist < MIN_DISTANCE_CLASSIFICATION_METERS or dist2 < MIN_DISTANCE_CLASSIFICATION_METERS:
            if print_logs:
                print(f"{bcolors.OKGREEN}Drone FOUND!{bcolors.ENDC}")

            result = CASE_MAVIC_PRO
    if scenario == "Scenario_1_4":
        az12, az21, dist = g.inv(xx, yy, alv_x, alv_y)
        az12, az21, dist2 = g.inv(xx, yy, arc_x, arc_y)

        if print_logs:
            print(f"({xx}, {yy});\n({arc_x}, {arc_y}); ({alv_x}, {alv_y})")
            print(f"1_alv: {dist}, 1_arc: {dist2}")

        if dist < MIN_DISTANCE_CLASSIFICATION_METERS or dist2 < MIN_DISTANCE_CLASSIFICATION_METERS:
            if print_logs:
                print(f"{bcolors.OKGREEN}Drone FOUND!{bcolors.ENDC}")

            result = CASE_MAVIC2
    if scenario == "Scenario_2_1":
        az12, az21, dist = g.inv(xx, yy, alv_x, alv_y)
        az12, az21, dist2 = g.inv(xx, yy, arc_x, arc_y)

        az12, az21, dist3 = g.inv(xx2, yy2, alv_x, alv_y)
        az12, az21, dist4 = g.inv(xx2, yy2, arc_x, arc_y)

        if print_logs:
            print(f"({xx}, {yy});\n({arc_x}, {arc_y}); ({alv_x}, {alv_y})")
            print(f"1_alv: {dist}, 1_arc: {dist2}, 2_alv: {dist3}, 2_arc: {dist4}")

        if dist < MIN_DISTANCE_CLASSIFICATION_METERS or dist2 < MIN_DISTANCE_CLASSIFICATION_METERS:
            if print_logs:
                print(f"{bcolors.OKGREEN}First drone FOUND!{bcolors.ENDC}")

            firstDrone = True

        if dist3 < MIN_DISTANCE_CLASSIFICATION_METERS or dist4 < MIN_DISTANCE_CLASSIFICATION_METERS:
            if print_logs:
                print(f"{bcolors.OKGREEN}Second drone FOUND!{bcolors.ENDC}")

            secondDrone = True

        if firstDrone and secondDrone:
            result = CASE_PHANTOM4PRO_MAVIC2
        else:
            if firstDrone:
                result = CASE_PHANTOM_4_PRO
            if secondDrone:
                result = CASE_MAVIC2
    if scenario == "Scenario_2_2":
        az12, az21, dist = g.inv(xx, yy, alv_x, alv_y)
        az12, az21, dist2 = g.inv(xx, yy, arc_x, arc_y)

        az12, az21, dist3 = g.inv(xx2, yy2, alv_x, alv_y)
        az12, az21, dist4 = g.inv(xx2, yy2, arc_x, arc_y)

        if print_logs:
            print(f"({xx}, {yy}); ({xx2}, {yy2});\n({arc_x}, {arc_y}); ({alv_x}, {alv_y})")
            print(f"1_alv: {dist}, 1_arc: {dist2}, 2_alv: {dist3}, 2_arc: {dist4}")

        if dist < MIN_DISTANCE_CLASSIFICATION_METERS or dist2 < MIN_DISTANCE_CLASSIFICATION_METERS:
            if print_logs:
                print(f"{bcolors.OKGREEN}First drone FOUND!{bcolors.ENDC}")
            firstDrone = True

        if dist3 < MIN_DISTANCE_CLASSIFICATION_METERS or dist4 < MIN_DISTANCE_CLASSIFICATION_METERS:
            if print_logs:
                print(f"{bcolors.OKGREEN}Second drone FOUND!{bcolors.ENDC}")
            secondDrone = True

        if firstDrone and secondDrone:
            result = CASE_PHANTOM4PRO_MAVICPRO
        else:
            if firstDrone:
                result = CASE_PHANTOM_4_PRO
            if secondDrone:
                result = CASE_MAVIC_PRO
    # no need for parrot A scenario

    if print_logs:
        print(f"ALVIRA: {val_alvira}, ARCUS: {val_arcus}, DIANA: {val_diana}, scenario: {scenario}, result: {result}")
        print("---")

    return result

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

MLP_CLASSIFIER_INPUT_COLUMNS = [
    'AlviraTracksTrackPosition_Altitude',
     'AlviraTracksTrackVelocity_Azimuth',
     'AlviraTracksTrackVelocity_Elevation',
     'AlviraTracksTrackVelocity_Speed',
     'AlviraTracksTrack_Classification',
     'AlviraTracksTrack_Reflection',
     'ArcusTracksTrackPosition_Altitude',
     'ArcusTracksTrackVelocity_Azimuth',
     'ArcusTracksTrackVelocity_Elevation',
     'ArcusTracksTrackVelocity_Speed',
     'ArcusTracksTrack_Classification',
     'ArcusTracksTrack_Reflection'
]

MLP_INPUT_COLUMNS = [
     'AlviraTracksTrackPosition_Altitude',
     'AlviraTracksTrackVelocity_Azimuth',
     'AlviraTracksTrackVelocity_Elevation',
     'AlviraTracksTrackVelocity_Speed',
     'AlviraTracksTrack_Classification',
     'AlviraTracksTrack_Reflection',
     'ArcusTracksTrackPosition_Altitude',
     'ArcusTracksTrackVelocity_Azimuth',
     'ArcusTracksTrackVelocity_Elevation',
     'ArcusTracksTrackVelocity_Speed',
     'ArcusTracksTrack_Classification',
     'ArcusTracksTrack_Reflection',
     'DianaTargetsTargetSignal_snr_dB',
     'DianaTargetsTargetSignal_bearing_deg',
     'DianaTargetsTargetSignal_range_m',
     'DianaTargetsTargetClassification_type',
     'reference_classification'
]

COLS_TO_STANDARDIZE = [
    'AlviraTracksTrackPosition_Altitude',
     'AlviraTracksTrackVelocity_Azimuth',
     'AlviraTracksTrackVelocity_Elevation',
     'AlviraTracksTrackVelocity_Speed',
     'AlviraTracksTrack_Reflection',
     'AlviraTracksTrack_Score',
     'ArcusTracksTrackPosition_Altitude',
     'ArcusTracksTrackVelocity_Azimuth',
     'ArcusTracksTrackVelocity_Elevation',
     'ArcusTracksTrackVelocity_Speed',
     'ArcusTracksTrack_Reflection',
     'ArcusTracksTrack_Score',
     'DianaTargetsTargetSignal_snr_dB',
     'DianaTargetsTargetSignal_bearing_deg',
     'DianaTargetsTargetSignal_range_m'
]