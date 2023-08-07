import csv
from datetime import datetime, timezone

def readCSV(fileName):
    file = open(fileName, "r")
    data = list(csv.DictReader(file, delimiter=","))
    file.close()

    return data

scens = ["Scenario_1_1", "Scenario_1_2_b", "Scenario_1_3", "Scenario_1_4", "Scenario_2_1", "Scenario_2_2", "Scenario_Parrot_a"]
logs = ["2020-09-29_14-10-56_v2.csv", #1_1
        "2020-09-29_15-55-18_v2.csv", #1_2_b
        "2020-09-29_15-18-15_v2.csv", #1_3
        "2020-09-29_15-43-48_v2.csv", #1_4
        "2020-09-30_10-38-45_v2.csv",  #"2020-09-30_10-38-42_v2.csv", #2_1
        "2020-09-30_11-01-07_v2.csv", #"2020-09-30_11-01-09_v2.csv", #2_2
        "2020_29-09-2020-13-58-31-Flight-Airdata_corrected.csv"] # parrot

test_file_index = 0

# for Parrots we need to build the timestamp index...
PARROT_INDEX = 6

def WriteHeader(writer):
    writer.writerow(['timestamp',
                              'scenario_name',
                              'latitude', 'longitude', 'altitude', 'speed',  # 1st drone, if detected
                              'latitude_2', 'longitude_2', 'altitude_2', 'speed_2',  # 2nd drone, if detected
                              'AlviraTracksTrackPosition_Latitude',
                              'AlviraTracksTrackPosition_Longitude',
                              'AlviraTracksTrackPosition_Altitude',
                              'AlviraTracksTrackVelocity_Azimuth',
                              'AlviraTracksTrackVelocity_Elevation',
                              'AlviraTracksTrackVelocity_Speed',
                              'AlviraTracksTrack_Classification',
                              'AlviraTracksTrack_Reflection',
                              'AlviraTracksTrack_Score',
                              'AlviraTracksTrack_Alarm',
                              'ArcusTracksTrackPosition_Latitude',
                              'ArcusTracksTrackPosition_Longitude',
                              'ArcusTracksTrackPosition_Altitude',
                              'ArcusTracksTrackVelocity_Azimuth',
                              'ArcusTracksTrackVelocity_Elevation',
                              'ArcusTracksTrackVelocity_Speed',
                              'ArcusTracksTrack_Classification',
                              'ArcusTracksTrack_Reflection',
                              'ArcusTracksTrack_Score',
                              'ArcusTracksTrack_Alarm',
                              'DianaTargetsTargetSignal_snr_dB',
                              'DianaTargetsTargetSignal_bearing_deg',
                              'DianaTargetsTargetSignal_range_m',
                              'DianaTargetsTargetClassification_type',
                              'VenusTriggerVenusName_isThreat',
                              'VenusTriggerLinkType_Uplink',
                              'VenusTrigger_VenusName',
                              'VenusTrigger_RadioId',
                              'VenusTrigge_LifeStatus',
                              'VenusTrigger_Frequency',
                              'VenusTrigge_FrequencyBand',
                              'VenusTrigger_OnAirStartTime',
                              'VenusTrigger_StopTime',
                              'VenusTrigger_Azimuth',
                              'VenusTrigger_Deviation'])

SCENARIO_BASE = '../datasets/merge/dataset_BUILD1_'
OUTPUT_NAME = '../datasets/merge/dataset_BUILD1.csv'

with open(OUTPUT_NAME, mode='w', newline='') as database_merged_file:
    database_writer = csv.writer(database_merged_file)
    WriteHeader(database_writer)

    for sss in scens:
        print(sss)
        SCENARIO_FILENAME = SCENARIO_BASE + sss + ".csv"
        SCENARIO_FILENAME = open(SCENARIO_FILENAME, mode='w', newline='')
        scenario_writer = csv.writer(SCENARIO_FILENAME)
        WriteHeader(scenario_writer)

        double_scenario = False
        second_test_file = None

        if sss == "Scenario_2_1" or sss == "Scenario_2_2":
            double_scenario = True

            if sss == "Scenario_2_1":
                second_test_file = readCSV(f"../datasets/train/{sss}/2020-09-30_10-38-42_v2.csv")
            else: # scenario 2_2
                second_test_file = readCSV(f"../datasets/train/{sss}/2020-09-30_11-01-09_v2.csv")

        venus = readCSV(f"../datasets/train/{sss}/VENUS_scenario.csv")
        diana = readCSV(f"../datasets/train/{sss}/DIANA_scenario.csv")
        arcus = readCSV(f"../datasets/train/{sss}/ARCUS_scenario.csv")
        alvira = readCSV(f"../datasets/train/{sss}/ALVIRA_scenario.csv")
        test_file = readCSV(f"../datasets/train/{sss}/{logs[test_file_index]}")

        merged = []
        last_timestamp = -1

        for i in test_file:
            #print("", i["latitude"], " --- ", i["longitude"], " --- ", i["altitude(m)"], " --- ", i["speed(mps)"], " --- ", i["timestamp"])
            #print(i["timestamp"])

            new_item = []

            # parrot has some different handle mechanism...
            if test_file_index != PARROT_INDEX:
                int_time = int(i["timestamp"])

                # sampling, only take the closer moment to the beginning of the second...
                int_integer = int(int_time / 1000)
                int_dec = int_time % 1000

                if int_integer == last_timestamp:
                    continue

                last_timestamp = int_integer

                #print("")
                #print(int_time)
                #print(int(int_time / 1000))
                #print(int_time % 1000)

                new_item.append(int_integer)
                new_item.append(sss)
                new_item.append(i["latitude"])
                new_item.append(i["longitude"])
                new_item.append(i["altitude(m)"])
                new_item.append(i["speed(mps)"])

                if double_scenario:
                    sec_found = False

                    for sec_elem in second_test_file:
                        int_time2 = int(sec_elem["timestamp"])
                        int_integer2 = int(int_time2 / 1000)
                        int_dec2 = int_time2 % 1000

                        if int_integer == int_integer2:
                            new_item.append(sec_elem["latitude"])
                            new_item.append(sec_elem["longitude"])

                            new_item.append(sec_elem["altitude(m)"])
                            new_item.append(sec_elem["speed(mps)"])

                            sec_found = True
                            break
                    if not sec_found:
                        new_item.append("0")
                        new_item.append("0")
                        new_item.append("0")
                        new_item.append("0")
                else:
                    new_item.append("0")
                    new_item.append("0")
                    new_item.append("0")
                    new_item.append("0")

            else:
                time_string = i["datetime(utc)"]
                int_integer = int(datetime.strptime(time_string, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())

                if int_integer == last_timestamp:
                    continue

                #print(time_string)
                #print(int_integer)
                last_timestamp = int_integer

                # for this wasted piece of device we need to convert speed and altitude...
                alt_meters = float(i["altitude(feet)"]) * 0.3048
                speed_mps = float(i["speed(mph)"]) * 0.44704

                new_item.append(int_integer)
                new_item.append(sss)
                new_item.append(i["latitude"])
                new_item.append(i["longitude"])

                #new_item.append(i["latitude"])
                #new_item.append(i["longitude"])
                new_item.append(alt_meters)
                new_item.append(speed_mps)
                new_item.append("0")
                new_item.append("0")
                new_item.append("0")
                new_item.append("0")

            alv_found = False
            for a in alvira:
                #print("")
                #print(a["total in seconds"])

                #alv_int = int(a["total in seconds"])

                if a["AlviraTracksTrack_Timestamp"] == "" or alv_found:
                    continue

                alvira_time = datetime.strptime(a["AlviraTracksTrack_Timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z").replace(tzinfo=timezone.utc)
                alv_int = int(datetime.strptime(a["AlviraTracksTrack_Timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z").replace(tzinfo=timezone.utc).timestamp())

                # 'cos actually 1,8 seconds are closer to 2!
                if alvira_time.microsecond / 1000 > 700:
                    alv_int += 1

                # AlviraTracksTrack_Timestamp
                # AlviraTracksTrackPosition_Latitude
                # AlviraTracksTrackPosition_Longitude
                # AlviraTracksTrackPosition_Altitude
                # AlviraTracksTrackVelocity_Azimuth
                # AlviraTracksTrackVelocity_Elevation
                # AlviraTracksTrackVelocity_Speed
                # AlviraTracksTrack_Classification
                # AlviraTracksTrack_Reflection
                # AlviraTracksTrack_Score
                # AlviraTracksTrack_Alarm

                if alv_int == int_integer and not alv_found:
                    #print("alv_found!!!")
                    #print(a["AlviraTracksTrack_Timestamp"])

                    if a["AlviraTracksTrack_Timestamp"] != "":
                        int_track = int(datetime.strptime(a["AlviraTracksTrack_Timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z").replace(tzinfo=timezone.utc).timestamp())
                        #print(int_track)
                        #print(int(a["total in seconds"]))

                    if (a["AlviraTracksTrackPosition_Latitude"] != "" or
                            a["AlviraTracksTrackPosition_Longitude"] != "" or
                            a["AlviraTracksTrackPosition_Altitude"] != "" or
                            a["AlviraTracksTrackVelocity_Azimuth"] != "" or
                            a["AlviraTracksTrackVelocity_Elevation"] != "" or
                            a["AlviraTracksTrackVelocity_Speed"] != "" or
                            a["AlviraTracksTrack_Classification"] != "" or
                            a["AlviraTracksTrack_Reflection"] != "" or
                            a["AlviraTracksTrack_Score"] != "" or
                            a["AlviraTracksTrack_Alarm"] != ""):
                        alv_found = True
                        #new_item.append(a["AlviraTracksTrackPosition_Latitude"])
                        #new_item.append(a["AlviraTracksTrackPosition_Longitude"])

                        new_item.append(a["AlviraTracksTrackPosition_Latitude"])
                        new_item.append(a["AlviraTracksTrackPosition_Longitude"])

                        new_item.append(a["AlviraTracksTrackPosition_Altitude"])
                        new_item.append(a["AlviraTracksTrackVelocity_Azimuth"])
                        new_item.append(a["AlviraTracksTrackVelocity_Elevation"])
                        new_item.append(a["AlviraTracksTrackVelocity_Speed"])
                        new_item.append(a["AlviraTracksTrack_Classification"])
                        new_item.append(a["AlviraTracksTrack_Reflection"])
                        new_item.append(a["AlviraTracksTrack_Score"])
                        new_item.append(a["AlviraTracksTrack_Alarm"])

            if not alv_found:
                new_item.append("0")
                new_item.append("0")
                new_item.append("0")
                new_item.append("0")
                new_item.append("0")
                new_item.append("0")
                new_item.append("VOID")
                new_item.append("0")
                new_item.append("0")
                new_item.append("Void")

            arc_found = False
            for arc in arcus:
                #print("")
                #print(arc)
                #print(arc["total in seconds"])

                arc_int = int(arc["total in seconds"])

                if arc["ArcusTracksTrack_Timestamp"] == "" or arc_found:
                    continue

                #print(arc["ArcusTracksTrack_Timestamp"])

                arcus_time = datetime.strptime(arc["ArcusTracksTrack_Timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z").replace(tzinfo=timezone.utc)
                arc_int = int(datetime.strptime(arc["ArcusTracksTrack_Timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z").replace(tzinfo=timezone.utc).timestamp())

                # 'cos actually 1,8 seconds are closer to 2!
                if arcus_time.microsecond / 1000 > 700:
                    arc_int += 1

                #ArcusTracksTrackPosition_Latitude
                #ArcusTracksTrackPosition_Longitude
                #ArcusTracksTrackPosition_Altitude
                #ArcusTracksTrackVelocity_Azimuth
                #ArcusTracksTrackVelocity_Elevation
                #ArcusTracksTrackVelocity_Speed
                #ArcusTracksTrack_Classification
                #ArcusTracksTrack_Reflection
                #ArcusTracksTrack_Score
                #ArcusTracksTrack_Alarm

                if arc_int == int_integer and not arc_found:
                    #print("arc_found!!!")

                    if(arc["ArcusTracksTrackPosition_Latitude"] != "" or
                            arc["ArcusTracksTrackPosition_Longitude"] != "" or
                            arc["ArcusTracksTrackPosition_Altitude"] != "" or
                            arc["ArcusTracksTrackVelocity_Azimuth"] != "" or
                            arc["ArcusTracksTrackVelocity_Elevation"] != "" or
                            arc["ArcusTracksTrackVelocity_Speed"] != "" or
                            arc["ArcusTracksTrack_Classification"] != "" or
                            arc["ArcusTracksTrack_Reflection"] != "" or
                            arc["ArcusTracksTrack_Score"] != "" or
                            arc["ArcusTracksTrack_Alarm"] != ""):
                        arc_found = True

                        #new_item.append(arc["ArcusTracksTrackPosition_Latitude"])
                        #new_item.append(arc["ArcusTracksTrackPosition_Longitude"])

                        new_item.append(arc["ArcusTracksTrackPosition_Latitude"])
                        new_item.append(arc["ArcusTracksTrackPosition_Longitude"])
                        new_item.append(arc["ArcusTracksTrackPosition_Altitude"])
                        new_item.append(arc["ArcusTracksTrackVelocity_Azimuth"])
                        new_item.append(arc["ArcusTracksTrackVelocity_Elevation"])
                        new_item.append(arc["ArcusTracksTrackVelocity_Speed"])
                        new_item.append(arc["ArcusTracksTrack_Classification"])
                        new_item.append(arc["ArcusTracksTrack_Reflection"])
                        new_item.append(arc["ArcusTracksTrack_Score"])
                        new_item.append(arc["ArcusTracksTrack_Alarm"])

            if not arc_found:
                new_item.append("0")
                new_item.append("0")
                new_item.append("0")
                new_item.append("0")
                new_item.append("0")
                new_item.append("0")
                new_item.append("VOID")
                new_item.append("0")
                new_item.append("0")
                new_item.append("Void")

            diana_found = False
            for d in diana:
                #print("")
                #print(d)
                #print(arc["total in seconds"])

                d_int = int(d["total in seconds"])

                if d_int == int_integer and not diana_found:
                    #print("d_found!!!")

                    # handle the case when information are in the following row with same timestamp!!!
                    # and of course the current row has no info inside...
                    if(d["DianaTargetsTargetSignal_snr_dB"] != "" or
                            d["DianaTargetsTargetSignal_bearing_deg"] != "" or
                            d["DianaTargetsTargetSignal_range_m"] != "" or
                            d["DianaTargetsTargetClassification_type"] != ""):
                        diana_found = True
                        new_item.append(d["DianaTargetsTargetSignal_snr_dB"])
                        new_item.append(d["DianaTargetsTargetSignal_bearing_deg"])
                        new_item.append(d["DianaTargetsTargetSignal_range_m"])
                        new_item.append(d["DianaTargetsTargetClassification_type"])

            if not diana_found:
                new_item.append("0")
                new_item.append("0")
                new_item.append("0")
                new_item.append("void")

            venus_found = False
            for v in venus:
                #print("")
                #print(v)
                #print(arc["total in seconds"])

                #VenusTriggerVenusName_isThreat
                #VenusTriggerLinkType_Uplink
                #VenusTrigger_VenusName
                #VenusTrigger_RadioId
                #VenusTrigge_LifeStatus
                #VenusTrigger_Frequency
                #VenusTrigge_FrequencyBand
                #VenusTrigger_OnAirStartTime
                #VenusTrigger_StopTime
                #VenusTrigger_Azimuth
                #VenusTrigger_Deviation

                v_int = int(v["total in seconds"])

                if v_int == int_integer and not venus_found:
                    #print("v_found!!!")

                    if(v["VenusTriggerVenusName_isThreat"] != "" or
                            v["VenusTriggerLinkType_Uplink"] != "" or
                            v["VenusTrigger_VenusName"] != "" or
                            v["VenusTrigger_RadioId"] != "" or
                            v["VenusTrigge_LifeStatus"] != "" or
                            v["VenusTrigger_Frequency"] != "" or
                            v["VenusTrigge_FrequencyBand"] != "" or
                            v["VenusTrigger_OnAirStartTime"] != "" or
                            v["VenusTrigger_StopTime"] != "" or
                            v["VenusTrigger_Azimuth"] != "" or
                            v["VenusTrigger_Deviation"] != ""):
                        venus_found = True

                        new_item.append(v["VenusTriggerVenusName_isThreat"])

                        if (v["VenusTriggerLinkType_Uplink"] == ""):
                            new_item.append("VOID")
                        else:
                            new_item.append(v["VenusTriggerLinkType_Uplink"])

                        if(v["VenusTrigger_VenusName"] == ""):
                            new_item.append("UNKNOWN")
                        else:
                            new_item.append(v["VenusTrigger_VenusName"])

                        new_item.append(v["VenusTrigger_RadioId"])
                        new_item.append(v["VenusTrigge_LifeStatus"])
                        new_item.append(v["VenusTrigger_Frequency"])
                        new_item.append(v["VenusTrigge_FrequencyBand"])
                        new_item.append(v["VenusTrigger_OnAirStartTime"])
                        new_item.append(v["VenusTrigger_StopTime"])

                        if (v["VenusTrigger_Azimuth"] == ""):
                            new_item.append("0")
                        else:
                            new_item.append(v["VenusTrigger_Azimuth"])

                        if (v["VenusTrigger_Deviation"] == ""):
                            new_item.append("0")
                        else:
                            new_item.append(v["VenusTrigger_Deviation"])

            if not venus_found:
                new_item.append("0")
                new_item.append("VOID")
                new_item.append("UNKNOWN")
                new_item.append("0")
                new_item.append("down")
                new_item.append("0")
                new_item.append("DOWN")
                new_item.append("0")
                new_item.append("0")
                new_item.append("0")
                new_item.append("0")

            merged.append(new_item)
            database_writer.writerow(new_item)
            scenario_writer.writerow(new_item)

        test_file_index += 1