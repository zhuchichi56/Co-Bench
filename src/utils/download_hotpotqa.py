"""
Download HotpotQA from HuggingFace and export to JSONL under src/data/.

Example:
  python src/utils/download_hotpotqa.py --num_samples 500
"""

import argparse

from download_common import (
    ensure_int,
    get_output_data_dir,
    human_mb,
    maybe_print,
    require_hf_datasets,
    write_jsonl,
)


def download_and_process_hotpotqa(
    num_samples: int = 500,
    split: str = "validation",
    *,
    verbose: bool = True,
    progress_every: int = 100,
    preview: int = 3,
) -> bool:
    try:
        num_samples = ensure_int("num_samples", num_samples, min_value=1)
        maybe_print(verbose, f"Downloading HotpotQA ({split}) from HuggingFace...")
        require_hf_datasets()
        from datasets import load_dataset

        dataset = load_dataset("hotpot_qa", "fullwiki", split=split)
        num_samples = min(num_samples, len(dataset))
        maybe_print(verbose, f"Dataset size: {len(dataset)}; processing first {num_samples}")

        output_data = []

        for idx, item in enumerate(dataset):
            if idx >= num_samples:
                break

            question = item['question']
            answer = item['answer']
            question_type = item['type']
            level = item['level']

            formatted_item = {
                "id": f"hotpotqa_{idx}",
                "instruction": f"Answer the following question.\n\nQuestion: {question}",
                "response": answer,
                "metadata": {
                    "type": question_type,
                    "level": level,
                    "dataset": "hotpotqa"
                }
            }

            output_data.append(formatted_item)

            if verbose and progress_every and (idx + 1) % progress_every == 0:
                print(f"processed={idx + 1}")

        output_file = get_output_data_dir() / f"hotpotqa_{num_samples}.jsonl"
        write_jsonl(output_file, output_data)
        maybe_print(verbose, f"\nSaved to: {output_file}")
        maybe_print(verbose, f"Rows: {len(output_data)}")
        try:
            maybe_print(verbose, f"File size: {human_mb(output_file.stat().st_size)}")
        except Exception:
            pass
        if verbose and preview:
            print("\nExample questions:")
            for i in range(min(preview, len(output_data))):
                item = output_data[i]
                inst = item.get("instruction", "")
                resp = item.get("response", "")
                meta = item.get("metadata", {})
                inst_short = inst[:150] + ("..." if len(inst) > 150 else "")
                print(f"\n{i+1}. {inst_short}")
                print(f"   answer: {resp}")
                print(f"   type: {meta.get('type')}, level: {meta.get('level')}")
        print(f"saved_jsonl={output_file} rows={len(output_data)}")

        return True

    except Exception as e:
        print(f"error: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download HotpotQA and export to JSONL under src/data/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/utils/download_hotpotqa.py -n 500
  python src/utils/download_hotpotqa.py -n 1000 -s train
        """
    )

    parser.add_argument(
        '-n', '--num_samples',
        type=int,
        default=500,
        help='Number of samples (default: 500)'
    )

    parser.add_argument(
        '-s', '--split',
        type=str,
        default='validation',
        choices=['train', 'validation'],
        help='Split (default: validation)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce prints (still prints saved_jsonl=... at the end).'
    )
    parser.add_argument(
        '--progress-every',
        type=int,
        default=100,
        help='Print progress every N samples (default: 100). Use 0 to disable.'
    )
    parser.add_argument(
        '--preview',
        type=int,
        default=3,
        help='How many samples to preview (default: 3). Use 0 to disable.'
    )

    args = parser.parse_args()

    success = download_and_process_hotpotqa(
        args.num_samples,
        args.split,
        verbose=not args.quiet,
        progress_every=args.progress_every,
        preview=args.preview,
    )

    if success:
        print("done")
    else:
        sys.exit(1)
