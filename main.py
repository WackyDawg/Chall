#!/usr/bin/env python3
"""
main.py — MirrorEye Wellbeing Intelligence System
Reply AI Agent Challenge 2026

CHALLENGE DAY USAGE:
  # Training runs (unlimited, check score each time):
  python main.py --level 1 --zip datasets/training_lev_1.zip
  python main.py --level 2 --zip datasets/training_lev_2.zip
  python main.py --level 3 --zip datasets/training_lev_3.zip

  # Final evaluation (ONE SHOT — add --eval flag):
  python main.py --level 1 --zip datasets/eval_lev_1.zip --eval
  python main.py --level 2 --zip datasets/eval_lev_2.zip --eval
  python main.py --level 3 --zip datasets/eval_lev_3.zip --eval

  # Use strong model if budget allows:
  python main.py --level 2 --zip datasets/training_lev_2.zip --strong

  # Override classification threshold (default 0.45):
  python main.py --level 1 --zip datasets/training_lev_1.zip --threshold 0.40
"""

import argparse
import os
import sys
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="MirrorEye — Wellbeing Intelligence System")
    parser.add_argument("--level", type=int, required=True, choices=[1, 2, 3],
                        help="Dataset level (1-3)")
    parser.add_argument("--zip", type=str, required=True,
                        help="Path to dataset zip file")
    parser.add_argument("--strong", action="store_true",
                        help="Use strong model (gpt-4o). Default: gpt-4o-mini")
    parser.add_argument("--eval", action="store_true",
                        help="Final evaluation mode — prompts confirmation before running")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override CLASSIFICATION_THRESHOLD (e.g. 0.40)")
    parser.add_argument("--output", type=str, default="output",
                        help="Output directory (default: output/)")
    args = parser.parse_args()

    # Override threshold if specified
    if args.threshold is not None:
        os.environ["CLASSIFICATION_THRESHOLD"] = str(args.threshold)
        print(f"[main] Threshold overridden → {args.threshold}")

    fast_model = os.getenv("FAST_MODEL") or os.getenv("OPENROUTER_MODEL", "gpt-4o-mini")
    strong_model = os.getenv("STRONG_MODEL") or os.getenv("OPENROUTER_MODEL", "gpt-4o")

    print(f"""
╔══════════════════════════════════════════════════════╗
║       MirrorEye — Wellbeing Intelligence System     ║
║       Reply AI Agent Challenge 2026                 ║
╚══════════════════════════════════════════════════════╝
  Level:     {args.level}
  Dataset:   {args.zip}
  Model:     {'STRONG (' + strong_model + ')' if args.strong else 'FAST (' + fast_model + ')'}
  Threshold: {os.getenv('CLASSIFICATION_THRESHOLD', '0.45')}
  Mode:      {'⚠️  FINAL EVALUATION' if args.eval else 'Training'}
""")

    if args.eval:
        print("⚠️  FINAL EVALUATION — You only get ONE submission per level!")
        print("    Confirm the output file looks correct before submitting.\n")
        confirm = input("Type 'yes' to proceed: ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            sys.exit(0)

    # Import here so env vars are set first
    from agents.coordinator import Coordinator

    coordinator = Coordinator(output_dir=args.output)
    session_id, output_file = coordinator.process_level(
        level=args.level,
        zip_path=args.zip,
        use_strong=args.strong,
    )

    print(f"""
╔══════════════════════════════════════════════════════╗
║                    DONE ✓                          ║
╚══════════════════════════════════════════════════════╝

  📄 Output file:  {output_file}
  🔑 Session ID:   {session_id}

  SUBMISSION CHECKLIST:
  {'[EVAL - FINAL]' if args.eval else '[TRAINING]'}
  □ Upload output file: {output_file}
  □ Paste Session ID:   {session_id}
  {'□ Upload source_code.zip (zip entire project folder)' if args.eval else '  (source code not needed for training)'}
""")


if __name__ == "__main__":
    main()