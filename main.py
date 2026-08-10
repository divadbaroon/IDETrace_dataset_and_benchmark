import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))  # repo root on path
import json
import yaml
import glob
import os
import random

from analysis.feature_extraction import compute_student_features

telemetry_events_file_path = "constants/telemetry_events.yaml"
data_dir = str(Path(__file__).resolve().parents[0] / "dataset" / "raw_telemetry")


def load_all_deployments(data_dir=data_dir):
    """Load and merge every deployment_*.json into one dict.
       Keys are namespaced 'deployment_N:session_id' so IDs can't collide across files.
       Each student is tagged with its 'deployment' for per-deployment reconciliation."""
    merged = {}
    paths = sorted(glob.glob(os.path.join(data_dir, "deployment_*.json")))
    for path in paths:
        with open(path, 'r') as f:
            deployment = json.load(f)
        if isinstance(deployment, dict) and "students" in deployment:
            # Wrapped raw source export (deployment_4.json, the intervention
            # session); read directly by run_preliminary_evaluation.py.
            continue
        dep_name = os.path.basename(path).replace(".json", "")
        for session_id, student in deployment.items():
            merged[f"{dep_name}:{session_id}"] = {**student, "deployment": dep_name}
    return merged


# Load all deployments once at module level
raw_telemetry = load_all_deployments()


def extract_telemetry_event_definitions():
    # Extract telemetry event constants
    with open(telemetry_events_file_path, 'r') as f:
        telemetry_events = yaml.safe_load(f)

    # Turn the lists of events into a dict of event count pairs
    telemetry_events_tracker = {}
    for events in telemetry_events.values():
        for event in events:
            telemetry_events_tracker[event] = 0
    return telemetry_events_tracker


def extract_student_data_from_deployment():
    # Create template of all event count pairs
    event_tracker_template = extract_telemetry_event_definitions()

    # Parse through all students across all deployments
    students = {}
    for session_id, student in raw_telemetry.items():

        # Ensure events are in chronological order
        student['events'].sort(key=lambda e: e['timestamp'])

        # Create fresh tracker dict per student
        event_tracker = dict(event_tracker_template)

        # Count total number of each event
        for event in student['events']:
            if event['type'] in event_tracker:
                event_tracker[event['type']] += 1

            # And event occurrences in each region
            if 'payload' in event and 'region' in event['payload']:
                region = event['payload']['region']
                if region in event_tracker:          # guard: unknown region won't crash the run
                    event_tracker[region] += 1

        # Add student to students if missing
        if student['student_name'] not in students:
            students[student['student_name']] = {}
        features = compute_student_features(events=student['events'], event_tracker=event_tracker)

        # Populate student tracker events per session
        students[student['student_name']][session_id] = {
            'events': event_tracker,
            'features': features
        }

    return students


def get_random_students(students, num_of_students=3):
    random_student_names = random.sample(list(students.keys()), num_of_students)

    random_students = {}
    for name in random_student_names:
        random_students[name] = students[name]

    return random_students