import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Segment an image with VASA.')
    parser.add_argument('--image', required=True)
    parser.add_argument('--query', required=True)
    parser.add_argument('--output', help='An empty output directory; defaults to an automatically named folder in outputs/.')
    parser.add_argument('--model', help='Overrides VASA_MODEL.')
    parser.add_argument('--base-url', help='Overrides VASA_BASE_URL.')
    parser.add_argument('--checkpoint', help='Local SAM3 checkpoint; otherwise use Hugging Face.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-rounds', type=int, default=20)
    parser.add_argument('--max-tokens', type=int, default=10000)
    parser.add_argument('--image-downscale', type=int, default=2)
    parser.add_argument('--reasoning-effort', choices=['low','medium','high'], help='Optional OpenRouter reasoning setting.')
    parser.add_argument('--no-trace', dest='save_trace', action='store_false', help='Discard the input copy, intermediate images, and conversation.')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / '.env', override=False)
    from src.inference import segment
    try:
        result = segment(**vars(args))
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
        parser.exit(1, f'VASA: {exc}\n')
    print(f"Saved mask and overlay to {result['output_dir']} ({result['termination']}, {result['vlm_calls']} VLM calls).")


if __name__ == '__main__':
    main()
