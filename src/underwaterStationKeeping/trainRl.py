from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from underwaterStationKeeping.rlBoxEnv import BoxStationParams, UnderwaterBoxStationEnv


ALGO_MAP = {"sac": SAC, "ppo": PPO, "td3": TD3}


def loadParams(configPath: str | None) -> BoxStationParams:
    if configPath is None:
        return BoxStationParams()
    with Path(configPath).open("r", encoding="utf-8") as configFile:
        configData = json.load(configFile)
    return BoxStationParams.fromDict(configData)


def makeEnv(seedValue: int, params: BoxStationParams):
    def initEnv():
        env = UnderwaterBoxStationEnv(params)
        env.reset(seed=seedValue)
        return Monitor(env)

    return initEnv


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="sac", choices=ALGO_MAP.keys())
    parser.add_argument("--total-timesteps", dest="totalTimesteps", type=int, default=300000)
    parser.add_argument("--run-name", dest="runName", default="rl_box_station_sac")
    parser.add_argument("--env-config", dest="envConfig", default=None)
    parser.add_argument("--load-model", dest="loadModel", default=None)
    parser.add_argument("--load-vecnormalize", dest="loadVecNormalize", default=None)
    parser.add_argument("--ent-coef", dest="entCoef", default="auto_0.1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parseArgs()
    runDir = Path("runs") / args.runName
    checkpointDir = runDir / "checkpoints"
    bestDir = runDir / "bestModel"
    tensorboardDir = runDir / "tensorboard"
    checkpointDir.mkdir(parents=True, exist_ok=True)
    bestDir.mkdir(parents=True, exist_ok=True)
    tensorboardDir.mkdir(parents=True, exist_ok=True)
    params = loadParams(args.envConfig)

    trainEnv = DummyVecEnv([makeEnv(args.seed, params)])
    evalEnv = DummyVecEnv([makeEnv(args.seed + 1000, params)])
    if args.loadVecNormalize:
        trainEnv = VecNormalize.load(args.loadVecNormalize, trainEnv)
        trainEnv.training = True
        trainEnv.norm_reward = True
        evalEnv = VecNormalize.load(args.loadVecNormalize, evalEnv)
    else:
        trainEnv = VecNormalize(trainEnv, norm_obs=True, norm_reward=True, clip_obs=10.0)
        evalEnv = VecNormalize(evalEnv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    evalEnv.training = False
    evalEnv.norm_reward = False

    callbacks = [
        CheckpointCallback(
            save_freq=50000,
            save_path=str(checkpointDir),
            name_prefix=f"{args.algo}_box_station",
            save_vecnormalize=True,
        ),
        EvalCallback(
            evalEnv,
            best_model_save_path=str(bestDir),
            log_path=str(runDir / "eval"),
            eval_freq=10000,
            n_eval_episodes=5,
            deterministic=True,
        ),
    ]

    algoClass = ALGO_MAP[args.algo]
    commonKwargs = {
        "policy": "MlpPolicy",
        "env": trainEnv,
        "verbose": 1,
        "tensorboard_log": str(tensorboardDir),
        "seed": args.seed,
        "device": args.device,
    }
    if args.loadModel:
        model = algoClass.load(
            args.loadModel,
            env=trainEnv,
            device=args.device,
            tensorboard_log=str(tensorboardDir),
        )
        print(f"loadedModel={args.loadModel}")
    elif args.algo == "sac":
        model = algoClass(
            **commonKwargs,
            learning_rate=3e-4,
            buffer_size=300000,
            learning_starts=5000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            train_freq=1,
            gradient_steps=1,
            ent_coef=args.entCoef,
            policy_kwargs={"net_arch": [128, 128]},
        )
    elif args.algo == "td3":
        model = algoClass(
            **commonKwargs,
            learning_rate=3e-4,
            buffer_size=300000,
            learning_starts=5000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            policy_kwargs={"net_arch": [128, 128]},
        )
    else:
        model = algoClass(
            **commonKwargs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
        )

    model.learn(total_timesteps=args.totalTimesteps, callback=callbacks, progress_bar=True)
    finalModelPath = runDir / "finalModel"
    model.save(finalModelPath)
    trainEnv.save(str(runDir / "vecNormalize.pkl"))
    print(f"savedModel={finalModelPath}.zip")
    print(f"savedVecNormalize={runDir / 'vecNormalize.pkl'}")


if __name__ == "__main__":
    main()
