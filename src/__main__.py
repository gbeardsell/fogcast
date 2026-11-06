"""Command-line entrypoint for a current baseline forecast."""
import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] != "forecast":
        print("Usage: python -m fogcast forecast --assets MODEL_BUNDLE --output DIRECTORY")
        raise SystemExit(0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 2)
    try:
        from .fogcast.forecast import main as forecast
        forecast(sys.argv[2:])
    except ImportError as error:
        raise SystemExit("Live inference dependencies are missing; install '.[live]'. " + str(error)) from error


if __name__ == "__main__":
    main()
