import os
from pathlib import Path
from instill.clients import init_pipeline_client


def get_pipeline_id(yaml_path):
    """Extract pipeline ID from YAML filename by removing extension"""
    return os.path.splitext(os.path.basename(yaml_path))[0]


def sync_recipes():
    # Initialize the pipeline client
    token = os.environ.get("INSTILL_API_TOKEN")
    pipeline = init_pipeline_client(api_token=token)

    # Get all YAML files in the current directory
    recipes_dir = Path(__file__).parent
    yaml_files = list(recipes_dir.glob("*.yaml"))

    print(f"Found {len(yaml_files)} YAML recipes to sync")

    try:
        for yaml_path in yaml_files:
            pipeline_id = get_pipeline_id(yaml_path)
            print(f"\nSyncing pipeline: {pipeline_id}")

            # Read the YAML file
            with open(yaml_path, 'r') as f:
                raw_recipe = f.read()

            # Update the pipeline
            try:
                pipeline.update_pipeline(
                    namespace_id="george_strong",
                    pipeline_id=pipeline_id,
                    description="",
                    raw_recipe=raw_recipe,
                )
                print(f"Successfully synced {pipeline_id}")
            except Exception as e:
                print(f"Error syncing {pipeline_id}: {str(e)}")

    finally:
        pipeline.close()


if __name__ == "__main__":
    sync_recipes()
