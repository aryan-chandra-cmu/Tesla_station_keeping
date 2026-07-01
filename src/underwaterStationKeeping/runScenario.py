from __future__ import annotations

import argparse
from pathlib import Path

from underwaterStationKeeping.simulation import (
    SimulationConfig,
    runSimulation,
    saveLog,
    summarizeLog,
)


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="mild", choices=["mild", "strong", "reversing"])
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--time-step", dest="timeStep", type=float, default=0.05)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parseArgs()
    outputPath = Path(args.output or f"results/{args.scenario}_run.csv")
    config = SimulationConfig(
        scenarioName=args.scenario,
        durationSec=args.duration,
        timeStep=args.timeStep,
    )
    logData = runSimulation(config)
    saveLog(logData, outputPath)
    summary = summarizeLog(logData)

    print(f"savedLog={outputPath}")
    for keyName, keyValue in summary.items():
        print(f"{keyName}={keyValue:.4f}")


if __name__ == "__main__":
    main()
