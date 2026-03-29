"""CLI for the R.A.G-Race-Router pulsed inference engine.

Usage:
    python -m engine --benchmark
    python -m engine --benchmark --runs 20
    python -m engine --status
    python -m engine --personality show
    python -m engine --personality reset
    python -m engine --config show
"""

import argparse
import json
import sys

from . import EngineConfig, RagRaceRouter


def main():
    parser = argparse.ArgumentParser(
        prog="rag-race-router",
        description="R.A.G-Race-Router — Adaptive Tri-Processor Inference Runtime",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run diagnostic benchmark across all devices",
    )
    parser.add_argument(
        "--runs", type=int, default=10,
        help="Number of benchmark iterations (default: 10)",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show current engine and hardware status",
    )
    parser.add_argument(
        "--personality", choices=["show", "reset", "update"],
        help="Show, reset, or update routing personality",
    )
    parser.add_argument(
        "--config", choices=["show", "default"],
        help="Show current config or print default config",
    )
    parser.add_argument(
        "--config-file", type=str,
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON (default: formatted text)",
    )
    parser.add_argument(
        "--temp-ceiling", type=float,
        help="GPU temperature ceiling in Celsius",
    )
    parser.add_argument(
        "--burst-ms", type=float,
        help="GPU burst duration in milliseconds",
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="Disable GPU execution",
    )

    args = parser.parse_args()

    # Build config
    if args.config_file:
        config = EngineConfig.from_file(args.config_file)
    else:
        config = EngineConfig()

    if args.temp_ceiling:
        config.temp_ceiling = args.temp_ceiling
    if args.burst_ms:
        config.gpu_burst_ms = args.burst_ms
    if args.no_gpu:
        config.gpu_enabled = False

    # Handle config commands
    if args.config:
        if args.config == "show":
            _output(config.to_dict(), args.json)
        elif args.config == "default":
            _output(EngineConfig().to_dict(), args.json)
        return

    # Handle personality commands
    if args.personality:
        from .personality import Personality
        p = Personality()
        if args.personality == "show":
            _output(p.show(), args.json)
        elif args.personality == "reset":
            import os
            from .personality import DB_PATH
            p.close()
            if DB_PATH.exists():
                os.remove(DB_PATH)
                print("Personality reset.")
            else:
                print("No personality data found.")
        elif args.personality == "update":
            p.update_rules()
            print("Routing rules updated.")
            _output(p.show(), args.json)
        p.close()
        return

    # Engine operations
    engine = RagRaceRouter(config=config)

    if args.status:
        engine.start()
        import time
        time.sleep(1)  # Let monitor collect a snapshot
        _output(engine.status(), args.json)
        engine.stop()
        return

    if args.benchmark:
        print(f"Running benchmark ({args.runs} iterations)...")
        engine.start()
        import time
        time.sleep(0.5)
        results = engine.benchmark(n=args.runs)
        _output(results, args.json)
        engine.stop()
        return

    # No command specified
    parser.print_help()


def _output(data: dict, as_json: bool):
    if as_json:
        print(json.dumps(data, indent=2))
    else:
        _pretty_print(data)


def _pretty_print(data: dict, indent: int = 0):
    prefix = "  " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            _pretty_print(value, indent + 1)
        elif isinstance(value, list):
            print(f"{prefix}{key}:")
            for item in value:
                if isinstance(item, dict):
                    _pretty_print(item, indent + 1)
                    print(f"{prefix}  ---")
                else:
                    print(f"{prefix}  - {item}")
        else:
            print(f"{prefix}{key}: {value}")


if __name__ == "__main__":
    main()
