"""Compare wall-clock vs active session durations."""
import json
import os
import yaml


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    manifest_path = os.path.join(root, 'manifest.yaml')

    if not os.path.exists(manifest_path):
        root = os.path.dirname(root)
        manifest_path = os.path.join(root, 'manifest.yaml')

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    EXCLUDE = {'MOUSE_MOVE', 'MOUSE_CLICK', 'WINDOW_RESIZE', 'PANEL_RESIZE',
            'TAB_STATE', 'SESSION_START', 'SESSION_END'}
    
    total_wall = 0
    total_active = 0
    total_students = 0

    for name, config in manifest['deployments'].items():
        if not config.get('enabled', True):
            continue
        path = os.path.join(root, 'dataset', config['raw_telemetry'])
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping {name}")
            continue

        with open(path) as f:
            raw = json.load(f)

        dep_wall = 0
        dep_active = 0
        dep_n = 0

        for sid, sdata in raw.items():
            events = sdata.get('events', [])
            if len(events) < 2:
                continue
            dep_n += 1

            timestamps = [e['timestamp'] for e in events]
            start_ts = min(timestamps)
            wall = (max(timestamps) - start_ts) / 1000 / 60
            dep_wall += wall

            last_meaningful_ts = start_ts
            for e in events:
                if e.get('type', '') not in EXCLUDE:
                    last_meaningful_ts = max(last_meaningful_ts, e.get('timestamp', 0))

            active = (last_meaningful_ts - start_ts) / 1000 / 60
            dep_active += active

        diff = dep_wall - dep_active
        pct = (diff / dep_wall * 100) if dep_wall > 0 else 0

        print(f"  {name}:")
        print(f"    Students: {dep_n}")
        print(f"    Wall clock:  {dep_wall:.1f} min")
        print(f"    Active:      {dep_active:.1f} min")
        print(f"    Difference:  {diff:.1f} min ({pct:.1f}%)")
        print()

        total_wall += dep_wall
        total_active += dep_active
        total_students += dep_n

    diff = total_wall - total_active
    pct = (diff / total_wall * 100) if total_wall > 0 else 0

    print(f"  {'═' * 50}")
    print(f"  TOTAL")
    print(f"  {'═' * 50}")
    print(f"  Students:     {total_students}")
    print(f"  Wall clock:   {total_wall:.1f} min ({total_wall/60:.1f} hours)")
    print(f"  Active:       {total_active:.1f} min ({total_active/60:.1f} hours)")
    print(f"  Difference:   {diff:.1f} min ({pct:.1f}%)")


if __name__ == '__main__':
    main()