"""
Simple script to run domain adaptation pipeline
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add parent directory to path to access utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domain_adaptation import DomainAdaptation
from utils.yaml_utils import load_yaml
from utils.logger_utils import setup_logger

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Domain Adaptation for Surgical Tool Pose Estimation")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration YAML file")
    parser.add_argument("--no-retrain", action="store_true", help="Skip retraining (only generate pseudo-labels)")

    args = parser.parse_args()

    try:
        # Validate config file exists
        if not Path(args.config).exists():
            print(f"Configuration file not found: {args.config}")
            sys.exit(1)
        
        # Load config and setup logging
        config = load_yaml(args.config)
        setup_logger(__name__, config)

        logger.info("Starting Domain Adaptation Pipeline...")
        logger.info(f"Using config: {args.config}")
        logger.info(f"Retraining: {'No' if args.no_retrain else 'Yes'}")
        logger.info("-" * 50)

        # Create domain adaptation instance
        da = DomainAdaptation(config_path=args.config)

        # Run domain adaptation
        results = da.run_domain_adaptation(retrain=not args.no_retrain)

        if len(results) > 3:
            logger.info("Domain adaptation completed successfully!")
            logger.info(f"Total iterations: {results['total_iterations']}")
            logger.info(f"Total pseudo-labels generated: {results['total_pseudo_labels']}")

            if results.get('final_model_path'):
                logger.info(f"Final refined model: {results['final_model_path']}")

            logger.info(f"Results saved in: {da.output_dir}")
            logger.info(f"Check 'overall_refinement_summary.json' for detailed results")
        else:
            logger.warning("Domain adaptation completed but no results generated")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Domain adaptation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()