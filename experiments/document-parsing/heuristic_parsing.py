import time
import json
import base64
from pathlib import Path
import os
from instill.clients import init_pipeline_client


def test_pdf_parsing():
    # Initialize the pipeline
    pipeline = init_pipeline_client(api_token=os.environ["INSTILL_API_TOKEN"])

    # Get all files from samples directory
    samples_dir = Path("experiments/document-parsing/samples")
    output_dir = Path("experiments/document-parsing/output/heuristic/markdown")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create timing stats directory
    stats_dir = Path("experiments/document-parsing/stats")
    stats_dir.mkdir(parents=True, exist_ok=True)
    timing_file = stats_dir / "heuristic_parsing_times.json"
    timing_data = {}

    # Get all PDF files
    supported_extensions = [".pdf"]
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

            # Read and encode the PDF file
            with open(sample_file, "rb") as f:
                pdf_bytes = f.read()
                pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')

            # Convert using Instill AI pipeline
            markdown_content = pipeline.trigger(
                namespace_id="george_strong",
                pipeline_id="standard-doc-to-markdown",
                data=[{"file": pdf_base64}]
            )['outputs'][0]['markdown']

            # Create output path with same name but .md extension
            output_path = output_dir / f"{sample_file.stem}.md"
            output_path.write_text(markdown_content)

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
            print(f"  {markdown_content[:200]}...")
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
    test_pdf_parsing()
