from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def findSettlingTime(timeValues: np.ndarray, errorValues: np.ndarray, thresholdValue: float) -> float | None:
    for sampleIndex in range(len(errorValues)):
        if np.all(errorValues[sampleIndex:] < thresholdValue):
            return float(timeValues[sampleIndex])
    return None


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/mild_run.csv")
    parser.add_argument("--output", default=None)
    parser.add_argument("--station-radius", dest="stationRadius", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    args = parseArgs()
    inputPath = Path(args.input)
    outputPath = Path(args.output or inputPath.with_suffix(".png"))
    outputPath.parent.mkdir(parents=True, exist_ok=True)

    data = np.genfromtxt(inputPath, delimiter=",", names=True)
    timeValues = data["timeSec"]
    positionError = data["positionErrorNorm"]
    yawErrorAbs = np.abs(data["yawError"])
    settlingTime = findSettlingTime(timeValues, positionError, args.stationRadius)
    finalPositionError = float(positionError[-1])
    finalYawErrorDeg = float(np.degrees(yawErrorAbs[-1]))
    maxForce = float(np.max(data["forceNorm"]))
    scenarioName = inputPath.stem.replace("_run", "").replace("_", " ").title()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.suptitle(f"{scenarioName} Current Station-Keeping", fontsize=15, fontweight="bold")

    stationCircle = Circle((0.0, 0.0), args.stationRadius, fill=False, linestyle="--", linewidth=1.5)
    axes[0, 0].add_patch(stationCircle)
    axes[0, 0].plot(data["positionX"], data["positionY"], linewidth=2.0, label="vehicle path")
    axes[0, 0].scatter(data["positionX"][0], data["positionY"][0], marker="o", s=45, label="start")
    axes[0, 0].scatter(data["positionX"][-1], data["positionY"][-1], marker="s", s=45, label="final")
    axes[0, 0].scatter([0.0], [0.0], marker="x", s=65, linewidths=2.0, label="target")
    axes[0, 0].set_title("Position")
    axes[0, 0].set_xlabel("x [m]")
    axes[0, 0].set_ylabel("y [m]")
    axes[0, 0].axis("equal")
    axes[0, 0].grid(True, alpha=0.35)
    axes[0, 0].legend(loc="best")

    axes[0, 1].plot(timeValues, positionError, linewidth=2.0, label="position error")
    axes[0, 1].plot(timeValues, yawErrorAbs, linewidth=1.8, label="yaw error")
    axes[0, 1].axhline(args.stationRadius, linestyle="--", linewidth=1.3, label="0.25 m tolerance")
    if settlingTime is not None:
        axes[0, 1].axvline(settlingTime, linestyle=":", linewidth=1.5, label=f"settled at {settlingTime:.1f}s")
    axes[0, 1].set_title("Tracking Error")
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].set_ylabel("error")
    axes[0, 1].grid(True, alpha=0.35)
    axes[0, 1].legend(loc="best")

    axes[1, 0].plot(timeValues, data["currentX"], linewidth=2.0, label="current x")
    axes[1, 0].plot(timeValues, data["currentY"], linewidth=2.0, label="current y")
    axes[1, 0].set_title("Ocean Current")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("m/s")
    axes[1, 0].grid(True, alpha=0.35)
    axes[1, 0].legend(loc="best")

    axes[1, 1].plot(timeValues, data["forceXBody"], linewidth=1.8, label="force x body")
    axes[1, 1].plot(timeValues, data["forceYBody"], linewidth=1.8, label="force y body")
    axes[1, 1].plot(timeValues, data["momentYaw"], linewidth=1.8, label="yaw moment")
    axes[1, 1].axhline(85.0, linestyle="--", linewidth=1.0, alpha=0.6, label="force limit")
    axes[1, 1].axhline(-85.0, linestyle="--", linewidth=1.0, alpha=0.6)
    axes[1, 1].set_title("Limited Control Commands")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("N or Nm")
    axes[1, 1].grid(True, alpha=0.35)
    axes[1, 1].legend(loc="best")

    summaryText = (
        f"Final position error: {finalPositionError:.3f} m\n"
        f"Final yaw error: {finalYawErrorDeg:.3f} deg\n"
        f"Max force norm: {maxForce:.1f} N"
    )
    if settlingTime is not None:
        summaryText += f"\nSettling time under {args.stationRadius:.2f} m: {settlingTime:.1f} s"
    fig.text(
        0.5,
        0.01,
        summaryText,
        ha="center",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.96])
    fig.savefig(outputPath, dpi=160)
    print(f"savedPlot={outputPath}")


if __name__ == "__main__":
    main()
