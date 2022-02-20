#!/usr/bin/python3

import argparse
import os
import csv
import numpy as np

minimum_time_window_threshold = 10.0
stationary_velocity_max_threshold = 0.5
moving_velocity_min_threshold = 5.0



def output_to_file(first_frame, last_frame, followed_frames, follower_frames, independent_frames, output_file_path, all_kinematic_variables, interagent_distance_variables, ttc_variables):
    if all_kinematic_variables:
        field_names = ["c0.a", "c0.v", "c0.p", "c1.a", "c1.v", "c1.p", "i0.a", "i0.v", "i0.p"]
    else:
        field_names = ["c0.a", "c1.a", "i0.a"]

    if interagent_distance_variables:
        field_names += ["c0-c1.d", "c0-i0.d", "c1-i0.d"]

    if ttc_variables:
        field_names += ["c0.ttc", "c1.ttc", "i0.ttc"]

    with open(output_file_path, "w") as output_file:
        csv_writer = csv.DictWriter(output_file, fieldnames=field_names)
        csv_writer.writeheader()

        followed_distance_travelled = None
        follower_distance_travelled = None
        independent_distance_travelled = None
        for followed_frame, follower_frame, independent_frame in zip(followed_frames, follower_frames, independent_frames):
            row = {}

            followed_acceleration = np.array((float(followed_frame["xAcceleration"]), float(followed_frame["yAcceleration"])))
            row["c0.a"] = np.linalg.norm(followed_acceleration)

            follower_acceleration = np.array((float(follower_frame["xAcceleration"]), float(follower_frame["yAcceleration"])))
            row["c1.a"] = np.linalg.norm(follower_acceleration)

            independent_acceleration = np.array((float(independent_frame["xAcceleration"]), float(independent_frame["yAcceleration"])))
            row["i0.a"] = np.linalg.norm(independent_acceleration)

            if all_kinematic_variables:
                followed_velocity = np.array((float(followed_frame["xVelocity"]), float(followed_frame["yVelocity"])))
                row["c0.v"] = np.linalg.norm(followed_velocity)

                follower_velocity = np.array((float(follower_frame["xVelocity"]), float(follower_frame["yVelocity"])))
                row["c1.v"] = np.linalg.norm(follower_velocity)

                independent_velocity = np.array((float(independent_frame["xVelocity"]), float(independent_frame["yVelocity"])))
                row["i0.v"] = np.linalg.norm(independent_velocity)

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

            if interagent_distance_variables:
                followed_position = np.array((float(followed_frame["x"]), float(followed_frame["y"])))
                follower_position = np.array((float(follower_frame["x"]), float(follower_frame["y"])))
                independent_position = np.array((float(independent_frame["x"]), float(independent_frame["y"])))

                row["c0-c1.d"] = np.linalg.norm(follower_position - followed_position)
                row["c0-i0.d"] = np.linalg.norm(independent_position - followed_position)
                row["c1-i0.d"] = np.linalg.norm(independent_position - follower_position)

            if ttc_variables:
                followed_ttc = float(followed_frame["ttc"])
                row["c0.ttc"] = followed_ttc

                follower_ttc = float(follower_frame["ttc"])
                row["c1.ttc"] = follower_ttc

                independent_ttc = float(independent_frame["ttc"])
                row["i0.ttc"] = independent_ttc

            csv_writer.writerow(row)



arg_parser = argparse.ArgumentParser(description="Extracts two agent convoy scenarios from the High-D dataset")
arg_parser.add_argument("input_directory_path")
arg_parser.add_argument("output_directory_path")
arg_parser.add_argument("--all-kinematic-variables", action="store_true")
arg_parser.add_argument("--interagent-distance-variables", action="store_true")
arg_parser.add_argument("--ttc-variables", action="store_true")
args = arg_parser.parse_args()

if not os.path.isdir(args.input_directory_path):
    raise ValueError(f"Input directory path {args.input_directory_path} is not a valid directory")

if not os.path.isdir(args.output_directory_path):
    raise ValueError(f"Output directory path {args.output_directory_path} is not a valid directory")

scene_count = int(len(os.listdir(args.input_directory_path)) / 4)

for i in range(1, scene_count + 1):
    print(f"Processing scene {i} of {scene_count}")

    recording_meta = None

    with open(os.path.join(args.input_directory_path, f"{i}_recordingMeta.csv"), "r") as recording_meta_file:
        recording_meta_reader = csv.DictReader(recording_meta_file)

        for row in recording_meta_reader:
            recording_meta = row
            break

    if recording_meta is None:
        print(f"Missing recording metadata for scene {i}")
        continue

    valid_tracks = {}
    valid_convoy_tracks = {}

    with open(os.path.join(args.input_directory_path, f"{i}_tracksMeta.csv"), "r") as tracks_meta_file:
        tracks_meta_reader = csv.DictReader(tracks_meta_file)

        for row in tracks_meta_reader:
            if int(row["numLaneChanges"]) == 0 \
            and float(row["numFrames"]) / float(recording_meta["frameRate"]) >= minimum_time_window_threshold:
                valid_tracks[int(row["id"])] = row

                if float(row["minXVelocity"]) <= stationary_velocity_max_threshold \
                and float(row["maxXVelocity"]) >= moving_velocity_min_threshold:
                    valid_convoy_tracks[int(row["id"])] = None

    if len(valid_convoy_tracks.keys()) == 0:
        continue

    track_frames = {}

    with open(os.path.join(args.input_directory_path, f"{i}_tracks.csv"), "r") as tracks_file:
        tracks_reader = csv.DictReader(tracks_file)

        current_id = None

        for row in tracks_reader:
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
            continue

        if valid_tracks.get(following_id) is None:
            continue

        following_metadata = valid_tracks[following_id]
        following_frames = track_frames[following_id]

        latest_initial_frame = max(float(metadata["initialFrame"]), float(following_metadata["initialFrame"]))
        earliest_final_frame = min(float(metadata["finalFrame"]), float(following_metadata["finalFrame"]))

        if (earliest_final_frame - latest_initial_frame) / float(recording_meta["frameRate"]) < minimum_time_window_threshold:
            continue

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

            output_to_file(
            updated_latest_initial_frame,
            updated_earliest_final_frame,
            updated_frames,
            updated_following_frames,
            updated_other_frames,
            os.path.join(args.output_directory_path, f"scene-{i}-{following_id}_follows_{valid_convoy_track}-{valid_track}_independent.csv"),
            args.all_kinematic_variables,
            args.interagent_distance_variables,
            args.ttc_variables)

            used_tracks[valid_track] = None
            used_tracks[valid_convoy_track] = None
            used_tracks[following_id] = None
            break
