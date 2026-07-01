from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from underwaterStationKeeping.rlBoxEnv import BoxStationParams, UnderwaterBoxStationEnv


ALGO_MAP = {"sac": SAC, "ppo": PPO, "td3": TD3}


def loadParams(configPath: str | None) -> BoxStationParams:
    if configPath is None:
        return BoxStationParams()
    with Path(configPath).open("r", encoding="utf-8") as configFile:
        configData = json.load(configFile)
    return BoxStationParams.fromDict(configData)


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="sac", choices=ALGO_MAP.keys())
    parser.add_argument("--model", required=True)
    parser.add_argument("--vecnormalize", required=True)
    parser.add_argument("--env-config", dest="envConfig", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="results/rl_policy_path.png")
    parser.add_argument("--csv-output", dest="csvOutput", default="results/rl_policy_path.csv")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def makeBoxCorners(params: BoxStationParams) -> np.ndarray:
    halfSize = np.array(
        [
            params.stationHalfSizeX,
            params.stationHalfSizeY,
            params.stationHalfSizeZ,
        ],
        dtype=float,
    )
    corners = []
    for xSign in [-1.0, 1.0]:
        for ySign in [-1.0, 1.0]:
            for zSign in [-1.0, 1.0]:
                corners.append([xSign * halfSize[0], ySign * halfSize[1], zSign * halfSize[2]])
    return np.asarray(corners, dtype=float)


def drawStationBox(axis, params: BoxStationParams) -> None:
    corners = makeBoxCorners(params)
    edgePairs = [
        (0, 1),
        (0, 2),
        (0, 4),
        (3, 1),
        (3, 2),
        (3, 7),
        (5, 1),
        (5, 4),
        (5, 7),
        (6, 2),
        (6, 4),
        (6, 7),
    ]
    for firstIndex, secondIndex in edgePairs:
        pointA = corners[firstIndex]
        pointB = corners[secondIndex]
        axis.plot(
            [pointA[0], pointB[0]],
            [pointA[1], pointB[1]],
            [pointA[2], pointB[2]],
            linestyle="--",
            linewidth=1.2,
            color="tab:green",
            alpha=0.85,
        )


def setAxesEqual(axis) -> None:
    xLimits = axis.get_xlim3d()
    yLimits = axis.get_ylim3d()
    zLimits = axis.get_zlim3d()
    xRange = abs(xLimits[1] - xLimits[0])
    yRange = abs(yLimits[1] - yLimits[0])
    zRange = abs(zLimits[1] - zLimits[0])
    maxRange = max(xRange, yRange, zRange)
    xMid = sum(xLimits) * 0.5
    yMid = sum(yLimits) * 0.5
    zMid = sum(zLimits) * 0.5
    axis.set_xlim3d([xMid - maxRange * 0.5, xMid + maxRange * 0.5])
    axis.set_ylim3d([yMid - maxRange * 0.5, yMid + maxRange * 0.5])
    axis.set_zlim3d([zMid - maxRange * 0.5, zMid + maxRange * 0.5])


def rolloutPolicy(args: argparse.Namespace, params: BoxStationParams) -> tuple[np.ndarray, float]:
    vecEnv = DummyVecEnv([lambda: UnderwaterBoxStationEnv(params)])
    vecEnv = VecNormalize.load(args.vecnormalize, vecEnv)
    vecEnv.training = False
    vecEnv.norm_reward = False
    vecEnv.seed(args.seed)
    model = ALGO_MAP[args.algo].load(args.model, env=vecEnv)

    obs = vecEnv.reset()
    done = False
    stepIndex = 0
    episodeReward = 0.0
    rows = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vecEnv.step(action)
        episodeReward += float(reward[0])
        done = bool(dones[0])
        info = infos[0]
        rows.append(
            [
                stepIndex * params.timeStep,
                info.get("positionX", np.nan),
                info.get("positionY", np.nan),
                info.get("positionZ", np.nan),
                info.get("positionErrorM", np.nan),
                info.get("speedMps", np.nan),
                float(info.get("insideBox", False)),
                info.get("currentX", np.nan),
                info.get("currentY", np.nan),
                info.get("currentZ", np.nan),
                info.get("forceX", np.nan),
                info.get("forceY", np.nan),
                info.get("forceZ", np.nan),
            ]
        )
        stepIndex += 1

    vecEnv.close()
    return np.asarray(rows, dtype=float), episodeReward


def saveRolloutCsv(rows: np.ndarray, outputPath: Path) -> None:
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    headerText = (
        "timeSec,positionX,positionY,positionZ,positionErrorM,speedMps,insideBox,"
        "currentX,currentY,currentZ,forceX,forceY,forceZ"
    )
    np.savetxt(outputPath, rows, delimiter=",", header=headerText, comments="")


def plotRollout(rows: np.ndarray, params: BoxStationParams, episodeReward: float, outputPath: Path, showPlot: bool) -> None:
    import matplotlib.pyplot as plt

    outputPath.parent.mkdir(parents=True, exist_ok=True)
    timeValues = rows[:, 0]
    finalError = rows[-1, 4]
    finalSpeed = rows[-1, 5]
    finalInsideBox = bool(rows[-1, 6])
    insideFraction = float(np.mean(rows[:, 6]))

    fig = plt.figure(figsize=(13, 8.5))
    pathAxis = fig.add_subplot(2, 2, 1, projection="3d")
    errorAxis = fig.add_subplot(2, 2, 2)
    currentAxis = fig.add_subplot(2, 2, 3)
    forceAxis = fig.add_subplot(2, 2, 4)

    drawStationBox(pathAxis, params)
    pathAxis.plot(rows[:, 1], rows[:, 2], rows[:, 3], linewidth=2.0, label="policy path")
    pathAxis.scatter(rows[0, 1], rows[0, 2], rows[0, 3], marker="o", s=50, label="start")
    pathAxis.scatter(rows[-1, 1], rows[-1, 2], rows[-1, 3], marker="s", s=50, label="final")
    pathAxis.scatter([0.0], [0.0], [0.0], marker="x", s=70, label="station center")
    pathAxis.set_title("3D Station Approach")
    pathAxis.set_xlabel("x [m]")
    pathAxis.set_ylabel("y [m]")
    pathAxis.set_zlabel("z [m]")
    pathAxis.legend(loc="best")
    setAxesEqual(pathAxis)

    errorAxis.plot(timeValues, rows[:, 4], linewidth=2.0, label="position error")
    errorAxis.plot(timeValues, rows[:, 5], linewidth=1.8, label="speed")
    errorAxis.axhline(
        np.linalg.norm(
            [
                params.stationHalfSizeX,
                params.stationHalfSizeY,
                params.stationHalfSizeZ,
            ]
        ),
        linestyle="--",
        linewidth=1.2,
        label="box diagonal radius",
    )
    errorAxis.set_title("Error and Speed")
    errorAxis.set_xlabel("time [s]")
    errorAxis.grid(True, alpha=0.35)
    errorAxis.legend(loc="best")

    currentAxis.plot(timeValues, rows[:, 7], label="current x")
    currentAxis.plot(timeValues, rows[:, 8], label="current y")
    currentAxis.plot(timeValues, rows[:, 9], label="current z")
    currentAxis.set_title("Current Velocity")
    currentAxis.set_xlabel("time [s]")
    currentAxis.set_ylabel("m/s")
    currentAxis.grid(True, alpha=0.35)
    currentAxis.legend(loc="best")

    forceAxis.plot(timeValues, rows[:, 10], label="force x")
    forceAxis.plot(timeValues, rows[:, 11], label="force y")
    forceAxis.plot(timeValues, rows[:, 12], label="force z")
    forceAxis.axhline(params.maxForceN, linestyle="--", linewidth=1.0, alpha=0.6, label="axis force limit")
    forceAxis.axhline(-params.maxForceN, linestyle="--", linewidth=1.0, alpha=0.6)
    forceAxis.set_title("Policy Thrust")
    forceAxis.set_xlabel("time [s]")
    forceAxis.set_ylabel("N")
    forceAxis.grid(True, alpha=0.35)
    forceAxis.legend(loc="best")

    summaryText = (
        f"Reward: {episodeReward:.1f} | Final error: {finalError:.3f} m | "
        f"Final speed: {finalSpeed:.3f} m/s | Inside box: {finalInsideBox} | "
        f"Inside fraction: {insideFraction:.2f}"
    )
    fig.suptitle("RL Underwater Station-Keeping Policy", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.015,
        summaryText,
        ha="center",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )
    fig.tight_layout(rect=[0.0, 0.06, 1.0, 0.95])
    fig.savefig(outputPath, dpi=160)
    if showPlot:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parseArgs()
    params = loadParams(args.envConfig)
    rows, episodeReward = rolloutPolicy(args, params)
    saveRolloutCsv(rows, Path(args.csvOutput))
    plotRollout(rows, params, episodeReward, Path(args.output), args.show)
    print(f"savedCsv={args.csvOutput}")
    print(f"savedPlot={args.output}")
    print(f"episodeReward={episodeReward:.3f}")
    print(f"finalPositionErrorM={rows[-1, 4]:.3f}")
    print(f"finalInsideBox={bool(rows[-1, 6])}")


if __name__ == "__main__":
    main()
