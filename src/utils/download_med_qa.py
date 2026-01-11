#!/usr/bin/env python3
"""
Download MedQA and export to JSONL under src/data/.

Notes:
- If you have local parquet files, set MEDQA_PARQUET_DIR to load them instead of Hub.

Examples:
  python src/utils/download_med_qa.py
  MEDQA_PARQUET_DIR=/path/to/medqa_parquet python src/utils/download_med_qa.py -n 2000
"""
import argparse
from pathlib import Path
import sys
import os

from download_common import (
    ensure_int,
    get_output_data_dir,
    human_mb,
    maybe_print,
    print_preview,
    require_hf_datasets,
    write_jsonl,
)

def download_and_extract_medqa(
    num_samples: int = 1000,
    *,
    verbose: bool = True,
    preview: int = 2,
) -> bool:
    """
    Download MedQA and convert to the project's JSONL format.

    Args:
        num_samples: Number of samples to export (default: 1000).
    """
    try:
        num_samples = ensure_int("num_samples", num_samples, min_value=1)

        require_hf_datasets()
        from datasets import load_dataset

        # Optional: local parquet directory via env var (avoid hardcoding machine paths).
        local_parquet_dir_env = os.environ.get("MEDQA_PARQUET_DIR", "").strip()
        local_parquet_dir = Path(local_parquet_dir_env) if local_parquet_dir_env else None
        split_name = "test"
        if local_parquet_dir and local_parquet_dir.exists():
            split_file_map = {
                "train": "train-00000-of-00001.parquet",
                "test": "test-00000-of-00001.parquet",
                "validation": "validation-00000-of-00001.parquet"
            }
            file_path = local_parquet_dir / split_file_map.get(split_name, "test-00000-of-00001.parquet")
            maybe_print(verbose, f"loading_local_parquet={file_path}")
            dataset = load_dataset("parquet", data_files=str(file_path), split="train")
        else:
            # Fallback: load from Hub.
            dataset = load_dataset("med_qa", "med_qa_en_bigbio_qa", split=split_name)

        total_available = len(dataset)
        num_samples = min(num_samples, total_available)
        maybe_print(verbose, f"Dataset size: {total_available}; processing: {num_samples}")

        converted_data = []
        for i in range(num_samples):
            item = dataset[i]

            question = item.get("question") or item.get("query") or item.get("text") or ""

            raw_options = item.get("options") if item.get("options") is not None else item.get("choices")
            choices = []
            option_labels = []

            if raw_options is None:
                raw_options = []

            if isinstance(raw_options, list) and len(raw_options) > 0 and isinstance(raw_options[0], dict):
                # Each option is a dict; try to read 'value'/'text' and 'key'.
                for opt in raw_options:
                    val = opt.get("value") or opt.get("text") or ""
                    key = opt.get("key")
                    choices.append(val)
                    option_labels.append(key)
            elif isinstance(raw_options, list):
                # List of strings.
                choices = [str(x) for x in raw_options]
                option_labels = [None] * len(choices)
            else:
                choices = []
                option_labels = []

            alpha_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
            for idx in range(len(choices)):
                if not option_labels[idx]:
                    option_labels[idx] = alpha_labels[idx] if idx < len(alpha_labels) else str(idx)

            correct_idx = None
            answer_idx = item.get("answer_idx")
            answer_field = item.get("answer")

            def label_to_index(lbl):
                if lbl is None:
                    return None
                lbls = str(lbl).strip()
                # Direct integer index.
                if lbls.isdigit():
                    return int(lbls)
                # Match letter labels.
                for j, lab in enumerate(option_labels):
                    if lab and lbls.lower() == str(lab).lower():
                        return j
                return None

            if isinstance(answer_idx, int):
                correct_idx = answer_idx
            elif isinstance(answer_idx, str):
                correct_idx = label_to_index(answer_idx)
            elif isinstance(answer_idx, list) and len(answer_idx) > 0:
                first = answer_idx[0]
                if isinstance(first, int):
                    correct_idx = first
                else:
                    correct_idx = label_to_index(first)
            elif isinstance(answer_field, list) and len(answer_field) > 0:
                first = answer_field[0]
                if isinstance(first, int):
                    correct_idx = first
                else:
                    correct_idx = label_to_index(first)
            elif isinstance(answer_field, (int, str)):
                correct_idx = label_to_index(answer_field)

            if correct_idx is None or correct_idx < 0 or correct_idx >= len(choices):
                maybe_print(verbose, f"Warning: skipping sample {i} due to invalid/missing answer")
                continue

            options_text = [f"{option_labels[idx]}. {choices[idx]}" for idx in range(len(choices))]

            instruction = (
                "The following is a multiple choice question about medical knowledge. "
                "Please select the correct answer.\n\n"
                f"Question: {question}\n\n"
                f"Options:\n" + "\n".join(options_text) + "\n\n"
                f"Please choose the correct answer from {', '.join(option_labels[:len(choices)])}."
            )

            correct_answer = option_labels[correct_idx]
            correct_text = choices[correct_idx]
            response = f"{correct_answer}. {correct_text}"

            converted_item = {
                "instruction": instruction,
                "response": response
            }

            converted_data.append(converted_item)

        output_file = get_output_data_dir() / "med_qa_1k.jsonl"
        write_jsonl(output_file, converted_data)
        maybe_print(verbose, f"\nSaved to: {output_file}")
        maybe_print(verbose, f"Rows: {len(converted_data)}")
        try:
            maybe_print(verbose, f"File size: {human_mb(output_file.stat().st_size)}")
        except Exception:
            pass
        if verbose and preview:
            print_preview(converted_data, n=preview, instruction_chars=200, response_chars=120)
        print(f"saved_jsonl={output_file} rows={len(converted_data)}")

        return True

    except Exception as e:
        print(f"error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download MedQA and export to JSONL under src/data/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/utils/download_med_qa.py
  python src/utils/download_med_qa.py -n 2000
        """
    )
    parser.add_argument(
        '-n', '--num-samples',
        type=int,
        default=1000,
        help='Number of samples (default: 1000)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce prints (still prints saved_jsonl=... at the end).'
    )
    parser.add_argument(
        '--preview',
        type=int,
        default=2,
        help='How many samples to preview (default: 2). Use 0 to disable.'
    )

    args = parser.parse_args()

    if args.num_samples <= 0:
        print("error: num_samples must be > 0")
        sys.exit(1)

    success = download_and_extract_medqa(
        args.num_samples,
        verbose=not args.quiet,
        preview=(0 if args.quiet else args.preview),
    )

    if success:
        print("done")
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
