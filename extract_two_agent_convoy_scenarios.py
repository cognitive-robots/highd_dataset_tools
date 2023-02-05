#!/usr/bin/python3

import argparse
import os
import shutil
import csv
import json
import numpy as np

minimum_time_window_threshold = 10.0
velocity_proportional_diff_threshold = 0.2



def output_to_file_csv(first_frame, last_frame, followed_frames, follower_frames, independent_frames, output_file_path, velocity_variables, all_kinematic_variables, interagent_distance_variables, ttc_variables):
    print(f"Generating output for {output_file_path}")

    if all_kinematic_variables:
        field_names = ["c0.a", "c0.v", "c0.p", "c1.a", "c1.v", "c1.p", "i0.a", "i0.v", "i0.p"]
    elif velocity_variables:
        field_names = ["c0.v", "c1.v", "i0.v"]
    else:
        field_names = ["c0.a", "c1.a", "i0.a"]

    if interagent_distance_variables:
        field_names += ["c0-c1.d", "c0-i0.d", "c1-i0.d"]

    if ttc_variables:
        field_names += ["c0.ttc", "c1.ttc", "i0.ttc"]

    with open(output_file_path, "w") as output_file:
        field_names = ["time_index"] + field_names
        csv_writer = csv.DictWriter(output_file, fieldnames=field_names)
        csv_writer.writeheader()

        followed_distance_travelled = None
        follower_distance_travelled = None
        independent_distance_travelled = None
        for i, (followed_frame, follower_frame, independent_frame) in enumerate(zip(followed_frames, follower_frames, independent_frames)):
            row = { "time_index": i }

            if all_kinematic_variables:
                row["c0.a"] = float(followed_frame["xAcceleration"])
                row["c1.a"] = float(follower_frame["xAcceleration"])
                row["i0.a"] = float(independent_frame["xAcceleration"])

                row["c0.v"] = float(followed_frame["xVelocity"])
                row["c1.v"] = float(follower_frame["xVelocity"])
                row["i0.v"] = float(independent_frame["xVelocity"])

                followed_position = np.array((float(followed_frame["x"]), float(followed_frame["y"])))
                if followed_distance_travelled is None:
                    followed_distance_travelled = 0
                else:
                    followed_distance_travelled += np.linalg.norm(followed_position - followed_previous_position)
                row["c0.p"] = followed_distance_travelled
                followed_previous_position = followed_position

                follower_position = np.array((float(follower_frame["x"]), float(follower_frame["y"])))
                if follower_distance_travelled is None:
                    follower_distance_travelled = 0
                else:
                    follower_distance_travelled += np.linalg.norm(follower_position - follower_previous_position)
                row["c1.p"] = follower_distance_travelled
                follower_previous_position = follower_position

                independent_position = np.array((float(independent_frame["x"]), float(independent_frame["y"])))
                if independent_distance_travelled is None:
                    independent_distance_travelled = 0
                else:
                    independent_distance_travelled += np.linalg.norm(independent_position - independent_previous_position)
                row["i0.p"] = independent_distance_travelled
                independent_previous_position = independent_position
            elif velocity_variables:
                row["c0.v"] = float(followed_frame["xVelocity"])
                row["c1.v"] = float(follower_frame["xVelocity"])
                row["i0.v"] = float(independent_frame["xVelocity"])
            else:
                row["c0.a"] = float(followed_frame["xAcceleration"])
                row["c1.a"] = float(follower_frame["xAcceleration"])
                row["i0.a"] = float(independent_frame["xAcceleration"])

            if interagent_distance_variables:
                followed_position = np.array((float(followed_frame["x"]), float(followed_frame["y"])))
                follower_position = np.array((float(follower_frame["x"]), float(follower_frame["y"])))
                independent_position = np.array((float(independent_frame["x"]), float(independent_frame["y"])))

                row["c0-c1.d"] = np.linalg.norm(follower_position - followed_position)
                row["c0-i0.d"] = np.linalg.norm(independent_position - followed_position)
                row["c1-i0.d"] = np.linalg.norm(independent_position - follower_position)

            if ttc_variables:
                row["c0.ttc"] = float(followed_frame["ttc"])
                row["c1.ttc"] = float(follower_frame["ttc"])
                row["i0.ttc"] = float(independent_frame["ttc"])

            csv_writer.writerow(row)



def output_to_file_json_meta(scene_id, convoy_head_id, convoy_tail_id, independent_id, output_file_path):
    print(f"Generating output for {output_file_path}")

    json_dict = {
        "scene_id": scene_id,
        "convoy_head_id": convoy_head_id,
        "convoy_tail_id": convoy_tail_id,
        "independent_id": independent_id
    }

    with open(output_file_path, "w") as output_file:
        json.dump(json_dict, output_file)


arg_parser = argparse.ArgumentParser(description="Extracts two agent convoy scenarios from the High-D dataset")
arg_parser.add_argument("input_directory_path")
arg_parser.add_argument("output_directory_path")

arg_parser.add_argument("--csv", action="store_true")
arg_parser.add_argument("--json-meta", action="store_true")

arg_parser.add_argument("--trimmed-scene-output-path")

arg_parser.add_argument("--velocity-variables", action="store_true")
arg_parser.add_argument("--all-kinematic-variables", action="store_true")
arg_parser.add_argument("--interagent-distance-variables", action="store_true")
arg_parser.add_argument("--ttc-variables", action="store_true")
args = arg_parser.parse_args()

if not os.path.isdir(args.input_directory_path):
    raise ValueError(f"Input directory path {args.input_directory_path} is not a valid directory")

if not os.path.isdir(args.output_directory_path):
    raise ValueError(f"Output directory path {args.output_directory_path} is not a valid directory")

if args.trimmed_scene_output_path is not None and not os.path.isdir(args.trimmed_scene_output_path):
    raise ValueError(f"Trimmed scene output directory path {args.trimmed_scene_output_path} is not a valid directory")

if not args.csv and not args.json_meta:
    raise ValueError("Please select either CSV or JSON meta output mode")

scene_count = int(len(os.listdir(args.input_directory_path)) / 4)

for i in range(1, scene_count + 1):
    print(f"Processing scene {i} of {scene_count}")

    recording_meta = None

    recording_meta_file_path = os.path.join(args.input_directory_path, f"{i}_recordingMeta.csv")

    with open(recording_meta_file_path, "r") as recording_meta_file:
        recording_meta_reader = csv.DictReader(recording_meta_file)

        for row in recording_meta_reader:
            recording_meta = row
            break

    if recording_meta is None:
        print(f"Missing recording metadata for scene {i}")
        continue

    tracks_meta_fieldnames = None
    valid_tracks = {}
    valid_convoy_tracks = {}

    tracks_meta_file_path = os.path.join(args.input_directory_path, f"{i}_tracksMeta.csv")

    with open(tracks_meta_file_path, "r") as tracks_meta_file:
        tracks_meta_reader = csv.DictReader(tracks_meta_file)

        for row in tracks_meta_reader:
            if tracks_meta_fieldnames is None:
                tracks_meta_fieldnames = list(row.keys())
            if int(row["numLaneChanges"]) == 0 \
            and float(row["numFrames"]) / float(recording_meta["frameRate"]) >= minimum_time_window_threshold:
                valid_tracks[int(row["id"])] = row

                if abs(float(row["maxXVelocity"]) - float(row["minXVelocity"])) / abs(float(row["maxXVelocity"])) >= velocity_proportional_diff_threshold:
                    valid_convoy_tracks[int(row["id"])] = -1

    print(f"Found {len(valid_tracks.keys())} valid agents and {len(valid_convoy_tracks.keys())} valid convoy agents")

    if len(valid_convoy_tracks.keys()) == 0:
        continue

    tracks_fieldnames = None
    track_frames = {}

    tracks_file_path = os.path.join(args.input_directory_path, f"{i}_tracks.csv")

    with open(tracks_file_path, "r") as tracks_file:
        tracks_reader = csv.DictReader(tracks_file)

        current_id = None

        for row in tracks_reader:
            if tracks_fieldnames is None:
                tracks_fieldnames = list(row.keys())
            if int(row["id"]) == current_id:
                frames.append(row)
            elif current_id is None:
                frames = [row]
                current_id = int(row["id"])
            else:
                track_frames[current_id] = frames
                frames = [row]
                current_id = int(row["id"])

        if current_id is not None:
            track_frames[current_id] = frames

    used_tracks = {}

    success_count = 0
    no_following_count = 0
    following_is_not_valid_convoy_count = 0
    too_short_before_other_count = 0
    no_suitable_other_count = 0
    for valid_convoy_track in valid_convoy_tracks.keys():
        following_id = None
        metadata = valid_tracks[valid_convoy_track]
        frames = track_frames[valid_convoy_track]

        for frame in frames:
            if following_id is None:
                lane_id = int(frame["laneId"])
                if int(frame["followingId"]) > 0:
                    following_id = int(frame["followingId"])
            else:
                if int(frame["followingId"]) > 0 and int(frame["followingId"]) != following_id:
                    following_id = None
                    break

        if following_id is None:
            no_following_count += 1
            continue

        if valid_convoy_tracks.get(following_id) is None:
            following_is_not_valid_convoy_count += 1
            continue

        following_metadata = valid_tracks[following_id]
        following_frames = track_frames[following_id]

        latest_initial_frame = max(float(metadata["initialFrame"]), float(following_metadata["initialFrame"]))
        earliest_final_frame = min(float(metadata["finalFrame"]), float(following_metadata["finalFrame"]))

        if (earliest_final_frame - latest_initial_frame) / float(recording_meta["frameRate"]) < minimum_time_window_threshold:
            too_short_before_other_count += 1
            continue

        found_suitable_other = False
        for valid_track in valid_tracks.keys():
            if used_tracks.get(valid_track) is not None:
                continue

            other_lane_id = None
            other_metadata = valid_tracks[valid_track]
            other_frames = track_frames[valid_track]

            updated_latest_initial_frame = max(latest_initial_frame, float(other_metadata["initialFrame"]))
            updated_earliest_final_frame = min(earliest_final_frame, float(other_metadata["finalFrame"]))

            if (updated_earliest_final_frame - updated_latest_initial_frame) / float(recording_meta["frameRate"]) < minimum_time_window_threshold:
                continue

            for frame in other_frames:
                if other_lane_id is None:
                    other_lane_id = int(frame["laneId"])
                    break

            if other_lane_id == lane_id:
                continue

            updated_frames = []
            for frame in frames:
                frame_num = int(frame["frame"])
                if frame_num >= updated_latest_initial_frame and frame_num <= updated_earliest_final_frame:
                    updated_frames.append(frame)

            updated_following_frames = []
            for frame in following_frames:
                frame_num = int(frame["frame"])
                if frame_num >= updated_latest_initial_frame and frame_num <= updated_earliest_final_frame:
                    updated_following_frames.append(frame)

            updated_other_frames = []
            for frame in other_frames:
                frame_num = int(frame["frame"])
                if frame_num >= updated_latest_initial_frame and frame_num <= updated_earliest_final_frame:
                    updated_other_frames.append(frame)

            if args.csv:
                output_to_file_csv(
                updated_latest_initial_frame,
                updated_earliest_final_frame,
                updated_frames,
                updated_following_frames,
                updated_other_frames,
                os.path.join(args.output_directory_path, f"scene-{i}-{following_id}_follows_{valid_convoy_track}-{valid_track}_independent.csv"),
                args.velocity_variables,
                args.all_kinematic_variables,
                args.interagent_distance_variables,
                args.ttc_variables)

            if args.json_meta:
                output_to_file_json_meta(
                i,
                valid_convoy_track,
                following_id,
                valid_track,
                os.path.join(args.output_directory_path, f"scene-{i}-{following_id}_follows_{valid_convoy_track}-{valid_track}_independent.json"))

            if args.trimmed_scene_output_path is not None:
                shutil.copy(recording_meta_file_path, os.path.join(args.trimmed_scene_output_path, f"scene-{i}-{following_id}_follows_{valid_convoy_track}-{valid_track}_independent-recordingMeta.csv"))

                present_ids = []

                with open(tracks_meta_file_path, "r") as tracks_meta_file:
                    tracks_meta_reader = csv.DictReader(tracks_meta_file)
                    with open(os.path.join(args.trimmed_scene_output_path, f"scene-{i}-{following_id}_follows_{valid_convoy_track}-{valid_track}_independent-tracksMeta.csv"), "w") as tracks_meta_output_file:
                        tracks_meta_csv_writer = csv.DictWriter(tracks_meta_output_file, fieldnames=tracks_meta_fieldnames)
                        tracks_meta_csv_writer.writeheader()
                        for row in tracks_meta_reader:
                            if float(row["initialFrame"]) < updated_earliest_final_frame \
                            and float(row["finalFrame"]) > updated_latest_initial_frame:
                                present_ids.append(row["id"])
                                tracks_meta_csv_writer.writerow(row)

                with open(tracks_file_path, "r") as tracks_file:
                    tracks_reader = csv.DictReader(tracks_file)
                    with open(os.path.join(args.trimmed_scene_output_path, f"scene-{i}-{following_id}_follows_{valid_convoy_track}-{valid_track}_independent-tracks.csv"), "w") as tracks_output_file:
                        tracks_csv_writer = csv.DictWriter(tracks_output_file, fieldnames=tracks_fieldnames)
                        tracks_csv_writer.writeheader()
                        for row in tracks_reader:
                            if row["id"] in present_ids:
                                tracks_csv_writer.writerow(row)



            found_suitable_other = True
            success_count += 1

            used_tracks[valid_track] = -1
            used_tracks[valid_convoy_track] = -1
            used_tracks[following_id] = -1
            break

        if not found_suitable_other:
            no_suitable_other_count += 1

    print(f"{success_count} Successes, {no_following_count} Failures due to no following agent, {following_is_not_valid_convoy_count} Failures due to following agent not being a valid convoy agent, {too_short_before_other_count} Failures due to too small of a time frame (prior to adding other agent), {no_suitable_other_count} Failures due to not being able to find suitable other agent")
