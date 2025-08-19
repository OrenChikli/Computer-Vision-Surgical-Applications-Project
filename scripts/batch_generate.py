#!/usr/bin/env python3
"""
Batch generation script for creating multiple datasets with different configurations.
Useful for experiments, ablation studies, or generating datasets with different parameters.
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path


def load_batch_config(batch_file):
    """Load batch configuration from YAML file."""
    try:
        import yaml
        with open(batch_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading batch config: {e}")
        return None

def create_config_from_template(base_config, overrides, output_path):
    """Create a new config file by applying overrides to base config."""
    try:
        import yaml
        
        # Load base config
        with open(base_config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Apply overrides
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(config, overrides)
        
        # Save new config
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        return True
    except Exception as e:
        print(f"Error creating config: {e}")
        return False

def run_generation(config_path, job_name):
    """Run synthetic data generation for a single configuration."""
    print(f"🚀 Starting job: {job_name}")
    start_time = time.time()
    
    try:
        # Run the generator
        cmd = [sys.executable, str(project_root / "synthetic_data_generator.py"), "--config", str(config_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Job {job_name} completed successfully in {duration/60:.1f} minutes")
            return True
        else:
            print(f"❌ Job {job_name} failed after {duration/60:.1f} minutes")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Job {job_name} crashed after {duration/60:.1f} minutes: {e}")
        return False

def run_batch_jobs(batch_config, base_config_path, output_dir):
    """Run all jobs defined in batch configuration."""
    jobs = batch_config.get('jobs', [])
    if not jobs:
        print("No jobs defined in batch configuration")
        return []
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    results = []
    total_start_time = time.time()
    
    print(f"🔧 Running {len(jobs)} batch jobs")
    print("=" * 60)
    
    for i, job in enumerate(jobs, 1):
        job_name = job.get('name', f'job_{i}')
        overrides = job.get('config', {})
        
        # Create config file for this job
        job_config_path = output_path / f"{job_name}_config.yaml"
        if not create_config_from_template(base_config_path, overrides, job_config_path):
            print(f"❌ Failed to create config for job {job_name}")
            results.append({'name': job_name, 'success': False, 'error': 'Config creation failed'})
            continue
        
        # Run generation
        print(f"\n[{i}/{len(jobs)}] Processing {job_name}...")
        success = run_generation(job_config_path, job_name)
        
        results.append({
            'name': job_name,
            'success': success,
            'config_path': str(job_config_path)
        })
    
    total_duration = time.time() - total_start_time
    
    # Summary
    successful_jobs = [r for r in results if r['success']]
    failed_jobs = [r for r in results if not r['success']]
    
    print(f"\n" + "=" * 60)
    print(f"📊 Batch Generation Summary")
    print(f"Total time: {total_duration/60:.1f} minutes")
    print(f"Successful jobs: {len(successful_jobs)}/{len(jobs)}")
    
    if successful_jobs:
        print("✅ Successful jobs:")
        for job in successful_jobs:
            print(f"  - {job['name']}")
    
    if failed_jobs:
        print("❌ Failed jobs:")
        for job in failed_jobs:
            print(f"  - {job['name']}")
    
    return results

def create_example_batch_config(output_path):
    """Create an example batch configuration file."""
    example_config = {
        'description': 'Example batch configuration for ablation study',
        'jobs': [
            {
                'name': 'baseline',
                'config': {
                    'num_images': 1000,
                    'motion_blur_prob': 0.0,
                    'occlusion_prob': 0.0,
                    'output_dir': 'output/ablation/baseline'
                }
            },
            {
                'name': 'with_motion_blur',
                'config': {
                    'num_images': 1000,
                    'motion_blur_prob': 0.3,
                    'occlusion_prob': 0.0,
                    'output_dir': 'output/ablation/motion_blur'
                }
            },
            {
                'name': 'with_occlusion',
                'config': {
                    'num_images': 1000,
                    'motion_blur_prob': 0.0,
                    'occlusion_prob': 0.3,
                    'output_dir': 'output/ablation/occlusion'
                }
            },
            {
                'name': 'with_both',
                'config': {
                    'num_images': 1000,
                    'motion_blur_prob': 0.3,
                    'occlusion_prob': 0.3,
                    'output_dir': 'output/ablation/both_effects'
                }
            }
        ]
    }
    
    try:
        import yaml
        with open(output_path, 'w') as f:
            yaml.dump(example_config, f, default_flow_style=False, indent=2)
        print(f"📝 Created example batch config: {output_path}")
        return True
    except Exception as e:
        print(f"Error creating example config: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Batch generation of synthetic surgical instrument datasets")
    parser.add_argument('--batch-config', required=True, help='YAML file defining batch jobs')
    parser.add_argument('--base-config', default='config.yaml', help='Base configuration file')
    parser.add_argument('--output-dir', default='batch_output', help='Directory for batch job outputs')
    parser.add_argument('--create-example', action='store_true', help='Create example batch configuration')
    parser.add_argument('--dry-run', action='store_true', help='Create configs but don\'t run generation')
    
    args = parser.parse_args()
    
    if args.create_example:
        example_path = "batch_config_example.yaml"
        if create_example_batch_config(example_path):
            print("✅ Example batch configuration created")
            print(f"Edit {example_path} and run with --batch-config {example_path}")
        return 0
    
    # Validate inputs
    if not Path(args.batch_config).exists():
        print(f"❌ Batch config file not found: {args.batch_config}")
        return 1
    
    if not Path(args.base_config).exists():
        print(f"❌ Base config file not found: {args.base_config}")
        return 1
    
    # Load batch configuration
    batch_config = load_batch_config(args.batch_config)
    if not batch_config:
        return 1
    
    print(f"📋 Loaded batch configuration: {args.batch_config}")
    print(f"🔧 Base configuration: {args.base_config}")
    print(f"📁 Output directory: {args.output_dir}")
    
    if args.dry_run:
        print("🔍 Dry run mode - creating configs only")
        # TODO: Implement dry run logic
        return 0
    
    # Run batch jobs
    results = run_batch_jobs(batch_config, args.base_config, args.output_dir)
    
    # Return appropriate exit code
    failed_jobs = [r for r in results if not r['success']]
    return 1 if failed_jobs else 0

if __name__ == "__main__":
    exit(main())
