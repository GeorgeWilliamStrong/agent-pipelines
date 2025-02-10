import os
import sys
from pathlib import Path
import time
import argparse
import asyncio
import concurrent.futures


async def run_experiment_async(script_path, description):
    """Run a single experiment script asynchronously and handle any errors."""
    print(f"\n{'='*80}")
    print(f"Running {description}")
    print(f"{'='*80}\n")

    start_time = time.time()

    try:
        # Change to the script's directory
        original_dir = os.getcwd()
        script_dir = os.path.dirname(script_path)
        os.chdir(script_dir)

        # Create process pool for running the script
        loop = asyncio.get_event_loop()
        with concurrent.futures.ProcessPoolExecutor() as pool:
            # Run the script in a separate process
            result = await loop.run_in_executor(
                pool,
                os.system,
                f"python {os.path.basename(script_path)}"
            )

        # Change back to original directory
        os.chdir(original_dir)

        if result == 0:
            print(f"\n✓ {description} completed successfully")
        else:
            print(f"\n✗ {description} failed with exit code {result}")

    except Exception as e:
        print(f"\n✗ Error running {description}: {str(e)}")
        return False
    finally:
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print(f"\n{'='*80}\n")

    return result == 0


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run document processing experiments')
    parser.add_argument(
        '--experiments',
        choices=['heuristic', 'hybrid', 'summarization', 'all'],
        nargs='+',
        default=['all'],
        help='Specify which experiments to run (default: all)'
    )
    parser.add_argument(
        '--async',
        dest='run_async',
        action='store_true',
        help='Run experiments asynchronously'
    )
    args = parser.parse_args()

    # Get the root directory (where this script is located)
    root_dir = Path(__file__).parent

    # Define all available experiments
    all_experiments = {
        'heuristic': {
            "script": root_dir / "document-parsing" / "heuristic_parsing.py",
            "description": "Heuristic Document Parsing Experiment"
        },
        'hybrid': {
            "script": root_dir / "document-parsing" / "hybrid_parsing.py",
            "description": "Hybrid Document Parsing Experiment"
        },
        'summarization': {
            "script": root_dir / "summarisation" / "markdown_summary.py",
            "description": "Document Summarization Experiment"
        }
    }

    # Determine which experiments to run
    experiments_to_run = []
    if 'all' in args.experiments:
        experiments_to_run = list(all_experiments.values())
    else:
        experiments_to_run = [all_experiments[exp] for exp in args.experiments]

    # Track overall success
    all_successful = True
    total_start_time = time.time()

    print("\nStarting experiments...")
    print(f"Running mode: {'Asynchronous' if args.run_async else 'Sequential'}")
    print(f"{'='*80}\n")

    if args.run_async:
        # Run experiments concurrently
        tasks = [run_experiment_async(exp["script"], exp["description"])
                for exp in experiments_to_run]
        results = await asyncio.gather(*tasks)
        all_successful = all(results)
    else:
        # Run experiments sequentially
        for exp in experiments_to_run:
            if not await run_experiment_async(exp["script"], exp["description"]):
                all_successful = False

    # Print final summary
    total_time = time.time() - total_start_time
    print("Experiments Summary:")
    print(f"{'='*80}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Overall status: {'✓ All experiments completed successfully' if all_successful else '✗ Some experiments failed'}")
    print(f"{'='*80}\n")

    # Exit with appropriate status code
    sys.exit(0 if all_successful else 1)

if __name__ == "__main__":
    asyncio.run(main())
