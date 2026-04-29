"""
Run benchmark per-deployment: trains on D1, tests on each deployment individually.
Saves results to separate JSON files for the generalization table.

Usage:
  cd tutortrace_dataset_and_benchmark
  python3 benchmark/run_per_deployment.py
"""

import yaml
import subprocess
import json
import os
import copy


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manifest_path = os.path.join(root, 'manifest.yaml')
    results_dir = os.path.join(root, 'benchmark', 'per_deployment_results')
    os.makedirs(results_dir, exist_ok=True)

    with open(manifest_path) as f:
        original_manifest = yaml.safe_load(f)

    # Find all deployments
    all_deployments = list(original_manifest['deployments'].keys())
    train_deployment = 'deployment_1'

    # Test each non-training deployment individually
    test_deployments = [d for d in all_deployments if d != train_deployment]

    print("=" * 60)
    print("  PER-DEPLOYMENT BENCHMARK")
    print("=" * 60)
    print(f"  Train: {train_deployment}")
    print(f"  Test deployments: {test_deployments}")
    print()

    for test_dep in test_deployments:
        print(f"\n{'=' * 60}")
        print(f"  RUNNING: {train_deployment} → {test_dep}")
        print(f"{'=' * 60}\n")

        # Build temporary manifest
        temp_manifest = copy.deepcopy(original_manifest)
        for dep_name, dep_config in temp_manifest['deployments'].items():
            if dep_name == train_deployment:
                dep_config['split'] = 'train'
                dep_config['enabled'] = True
            elif dep_name == test_dep:
                dep_config['split'] = 'test'
                dep_config['enabled'] = True
            else:
                dep_config['enabled'] = False

        # Write temporary manifest
        temp_manifest_path = os.path.join(root, 'manifest_temp.yaml')
        with open(temp_manifest_path, 'w') as f:
            yaml.dump(temp_manifest, f, default_flow_style=False)

        # Backup original and swap in temp
        backup_path = os.path.join(root, 'manifest_backup.yaml')
        os.rename(manifest_path, backup_path)
        os.rename(temp_manifest_path, manifest_path)

        try:
            # Run benchmark
            result = subprocess.run(
                ['python3', os.path.join(root, 'benchmark', 'run_benchmark.py')],
                cwd=root,
                capture_output=True,
                text=True,
            )

            # Save output
            output_path = os.path.join(results_dir, f'{test_dep}_output.txt')
            with open(output_path, 'w') as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write('\n\nSTDERR:\n')
                    f.write(result.stderr)

            # Copy results.json
            src_results = os.path.join(root, 'benchmark', 'results.json')
            if os.path.exists(src_results):
                dst_results = os.path.join(results_dir, f'{test_dep}_results.json')
                with open(src_results) as f:
                    data = json.load(f)
                with open(dst_results, 'w') as f:
                    json.dump(data, f, indent=2)

            print(f"  ✓ {test_dep} complete — saved to {results_dir}/")

            if result.returncode != 0:
                print(f"  WARNING: benchmark returned code {result.returncode}")
                if result.stderr:
                    print(f"  STDERR: {result.stderr[:500]}")

        finally:
            # Always restore original manifest
            os.rename(manifest_path, temp_manifest_path)
            os.rename(backup_path, manifest_path)
            os.remove(temp_manifest_path)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  ALL RUNS COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Results saved to: {results_dir}/")
    print(f"  Files:")
    for f in sorted(os.listdir(results_dir)):
        print(f"    {f}")


if __name__ == '__main__':
    main()