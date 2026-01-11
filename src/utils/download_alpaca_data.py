#!/usr/bin/env python3
"""
Download Alpaca from HuggingFace and export to JSONL under src/data/.

Examples:
  python src/utils/download_alpaca_data.py 10   # 10k rows
  python src/utils/download_alpaca_data.py 52   # full (~52k)
"""

import argparse
import sys

from download_common import (
    ensure_int,
    get_output_data_dir,
    human_mb,
    maybe_print,
    print_preview,
    require_hf_datasets,
    write_jsonl,
)


def download_and_extract_alpaca(num_k: int, *, verbose: bool = True, preview: int = 3) -> bool:
    """
    Download Alpaca from HuggingFace and export a subset.

    Args:
        num_k: Number of rows in thousands.
    """
    num_samples = ensure_int("num_k", num_k, min_value=1) * 1000
    maybe_print(verbose, "Downloading Alpaca from HuggingFace...")
    maybe_print(verbose, f"Target: {num_samples} rows")

    try:
        require_hf_datasets()
        from datasets import load_dataset

        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        total_available = len(dataset)
        num_samples = min(num_samples, total_available)
        maybe_print(verbose, f"Dataset size: {total_available}")
        if num_samples < ensure_int("num_k", num_k, min_value=1) * 1000:
            maybe_print(verbose, f"Warning: requested more than available, using {num_samples}")

        extracted_data = []
        for i in range(num_samples):
            item = dataset[i]

            converted_item = {
                "instruction": item.get("instruction", ""),
                "response": item.get("output", "")
            }

            if item.get("input", "").strip():
                converted_item["instruction"] = f"{item['instruction']}\n{item['input']}"

            extracted_data.append(converted_item)

        output_file = get_output_data_dir() / f"alpaca_{num_k}k.jsonl"
        write_jsonl(output_file, extracted_data)

        maybe_print(verbose, f"\nSaved to: {output_file}")
        maybe_print(verbose, f"Rows: {len(extracted_data)}")
        try:
            maybe_print(verbose, f"File size: {human_mb(output_file.stat().st_size)}")
        except Exception:
            pass
        if preview:
            print_preview(extracted_data, n=preview, instruction_chars=100, response_chars=100)
        print(f"saved_jsonl={output_file} rows={len(extracted_data)}")

        return True

    except Exception as e:
        print(f"error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download Alpaca and export to JSONL under src/data/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/utils/download_alpaca_data.py 10
  python src/utils/download_alpaca_data.py 52
        """
    )
    parser.add_argument(
        'num_k',
        type=int,
        help='Number of rows in thousands (e.g. 10 means 10k)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce prints (still prints saved_jsonl=... at the end).'
    )
    parser.add_argument(
        '--preview',
        type=int,
        default=3,
        help='How many samples to preview (default: 3). Use 0 to disable.'
    )

    args = parser.parse_args()

    if args.num_k <= 0:
        print("error: num_k must be > 0")
        sys.exit(1)

    success = download_and_extract_alpaca(args.num_k, verbose=not args.quiet, preview=args.preview)

    if success:
        print("done")
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()

