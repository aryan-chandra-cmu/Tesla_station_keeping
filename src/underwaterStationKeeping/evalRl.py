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
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output", default="results/rl_eval.csv")
    return parser.parse_args()


def main() -> None:
    args = parseArgs()
    outputPath = Path(args.output)
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    params = loadParams(args.envConfig)

    vecEnv = DummyVecEnv([lambda: UnderwaterBoxStationEnv(params)])
    vecEnv = VecNormalize.load(args.vecnormalize, vecEnv)
    vecEnv.training = False
    vecEnv.norm_reward = False
    model = ALGO_MAP[args.algo].load(args.model, env=vecEnv)

    rows = []
    for episodeIndex in range(args.episodes):
        obs = vecEnv.reset()
        done = False
        episodeReward = 0.0
        stepIndex = 0
        finalInfo = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vecEnv.step(action)
            episodeReward += float(reward[0])
            done = bool(dones[0])
            finalInfo = infos[0]
            rows.append(
                [
                    episodeIndex,
                    stepIndex,
                    episodeReward,
                    finalInfo.get("positionErrorM", np.nan),
                    finalInfo.get("speedMps", np.nan),
                    float(finalInfo.get("insideBox", False)),
                    finalInfo.get("positionX", np.nan),
                    finalInfo.get("positionY", np.nan),
                    finalInfo.get("positionZ", np.nan),
                ]
            )
            stepIndex += 1
        print(
            f"episode={episodeIndex} reward={episodeReward:.3f} "
            f"positionErrorM={finalInfo.get('positionErrorM', np.nan):.3f} "
            f"insideBox={finalInfo.get('insideBox', False)}"
        )

    headerText = (
        "episodeIndex,stepIndex,episodeReward,positionErrorM,speedMps,"
        "insideBox,positionX,positionY,positionZ"
    )
    np.savetxt(outputPath, np.asarray(rows), delimiter=",", header=headerText, comments="")
    print(f"savedEval={outputPath}")


if __name__ == "__main__":
    main()
