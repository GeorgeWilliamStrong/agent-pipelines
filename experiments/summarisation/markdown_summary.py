import time
import json
from pathlib import Path
import os
from instill.clients import init_pipeline_client


def test_summarisation():
    # Initialize the pipeline
    pipeline = init_pipeline_client(api_token=os.environ["INSTILL_API_TOKEN"])

    # Get all files from samples directory
    samples_dir = Path("samples")
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create timing stats directory
    stats_dir = Path("stats")
    stats_dir.mkdir(parents=True, exist_ok=True)
    timing_file = stats_dir / "summarisation_times.json"
    timing_data = {}

    # Get all Markdown files
    supported_extensions = [".md"]
    sample_files = [
        f for f in samples_dir.iterdir()
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]

    if not sample_files:
        print(f"No supported documents found in {samples_dir}")
        return

    print(f"Found {len(sample_files)} documents to process")
    print("-" * 50)

    # Process each file
    for sample_file in sample_files:
        try:
            start_time = time.time()

            # Read the markdown file
            with open(sample_file, "r", encoding="utf-8") as f:
                markdown_content = f.read()

            # Convert using Instill AI pipeline
            summary_content = pipeline.trigger(
                namespace_id="george_strong",
                pipeline_id="agent-summary",
                data=[{"context": markdown_content, "file-type": "document"}]
            )['outputs'][0]

            # Get summary from either long or short text field
            final_summary = (
                summary_content.get("summary-from-long-text") or 
                summary_content.get("summary-from-short-text", "No summary generated")
            )

            # Create output path with same name but add "-summary" suffix
            output_path = output_dir / f"{sample_file.stem}-summary.md"
            output_path.write_text(final_summary)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Update timing data
            if sample_file.name not in timing_data:
                timing_data[sample_file.name] = []
            timing_data[sample_file.name].append({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'processing_time': processing_time,
                'file_size': sample_file.stat().st_size,
                'success': True
            })

            print(f"✓ Successfully converted {sample_file.name}")
            print(f"  Processing time: {processing_time:.2f} seconds")
            print(f"  Output saved to {output_path}")
            print(f"  Preview (first 200 chars):")
            print(f"  {final_summary[:200]}...")
            print("-" * 50)

        except Exception as e:
            # Record failed attempt
            if sample_file.name not in timing_data:
                timing_data[sample_file.name] = []
            timing_data[sample_file.name].append({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'error': str(e),
                'file_size': sample_file.stat().st_size,
                'success': False
            })

            print(f"✗ Failed to convert {sample_file.name}")
            print(f"  Error: {str(e)}")
            print("-" * 50)

    # Save timing data
    timing_file.write_text(json.dumps(timing_data, indent=2))


if __name__ == "__main__":
    test_summarisation()
