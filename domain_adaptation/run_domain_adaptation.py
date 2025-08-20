"""
Simple script to run domain adaptation pipeline
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path to access utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domain_adaptation import DomainAdaptation


def main():
    parser = argparse.ArgumentParser(description="Run Domain Adaptation for Surgical Tool Pose Estimation")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration YAML file")
    parser.add_argument("--no-retrain", action="store_true", help="Skip retraining (only generate pseudo-labels)")

    args = parser.parse_args()

    try:
        # Validate config file exists
        if not Path(args.config).exists():
            print(f"❌ Configuration file not found: {args.config}")
            sys.exit(1)

        print("🚀 Starting Domain Adaptation Pipeline...")
        print(f"📋 Using config: {args.config}")
        print(f"🔄 Retraining: {'No' if args.no_retrain else 'Yes'}")
        print("-" * 50)

        # Create domain adaptation instance
        da = DomainAdaptation(config_path=args.config)

        # Run domain adaptation
        results = da.run_domain_adaptation(retrain=not args.no_retrain)

        if len(results) > 3:
            print("\n🎉 Domain adaptation completed successfully!")
            print(f"📊 Total iterations: {results['total_iterations']}")
            print(f"📈 Total pseudo-labels generated: {results['total_pseudo_labels']}")

            if results.get('final_model_path'):
                print(f"🎯 Final refined model: {results['final_model_path']}")

            print(f"📄 Results saved in: {da.output_dir}")
            print(f"📋 Check 'overall_refinement_summary.json' for detailed results")
        else:
            print("⚠️ Domain adaptation completed but no results generated")

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Domain adaptation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()