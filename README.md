# Underwater Vehicle Station-Keeping

This project simulates a planar underwater vehicle holding position and heading under time-varying ocean currents. The controller is a PID station-keeping controller with force saturation and thrust slew-rate limits. As an extension, a RL controller was also developed for the same task.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
```

For Conda:

```powershell
conda create -n underwater-station python=3.11 -y
conda activate underwater-station
python -m pip install -e .
```

## Run One Scenario

```powershell
python -m underwaterStationKeeping.runScenario --scenario mild --output results/mild_run.csv
```

Available scenarios:

```text
mild
strong
reversing
```

## Generate Plots

```powershell
python -m underwaterStationKeeping.plotResults --input results/mild_run.csv --output results/mild_run.png
```

## Run All Scenarios

```powershell
.\scripts\run_all.ps1
```


## Model

The vehicle state is:

```text
x, y, yaw, surge velocity, sway velocity, yaw rate
```

The control command is:

```text
body x force, body y force, yaw moment
```

Ocean current acts as a world-frame velocity disturbance. Thruster commands are limited by maximum force, maximum moment, force slew rate, and moment slew rate.

## Expected Output

Each run saves a CSV file with state, current, control command, and error values. The plot script creates a four-panel image:

```text
vehicle path
tracking errors
current disturbance
control commands
```

## Suggested Report Use

Use `reports/report.md` as the starting report. Add generated plots from the `results` folder.

## Optional RL Version

The RL version models the station as a 3D box around the target. The observation contains normalized position error, vehicle velocity, current velocity, and previous thrust command. The action is normalized thrust in world `x`, `y`, and `z`.

Install RL dependencies:

```powershell
python -m pip install -r requirements-rl.txt
```

Train SAC:

```powershell
python -m underwaterStationKeeping.trainRl --algo sac --total-timesteps 300000 --run-name rl_box_station_sac
```

Evaluate:

```powershell
python -m underwaterStationKeeping.evalRl --algo sac --model runs/rl_box_station_sac/finalModel.zip --vecnormalize runs/rl_box_station_sac/vecNormalize.pkl --episodes 5 --output results/rl_eval.csv
```

PowerShell helper:

```powershell
.\scripts\train_rl.ps1
```

## Optional RL Curriculum

Because the full problem was too much to train in a single go,the curriculum starts with a larger station box, small initial offsets, and weak current. It then increases the initial offset, tightens the box, and restores the full current profile.

```powershell
.\scripts\train_rl_curriculum.ps1
```

Manual stages:

```powershell
python -m underwaterStationKeeping.trainRl --algo sac --env-config configs/rl_stage1_hold.json --total-timesteps 150000 --run-name rl_curr_stage1_hold --ent-coef auto_0.1
python -m underwaterStationKeeping.trainRl --algo sac --env-config configs/rl_stage2_recover.json --total-timesteps 200000 --run-name rl_curr_stage2_recover --load-model runs/rl_curr_stage1_hold/finalModel.zip --load-vecnormalize runs/rl_curr_stage1_hold/vecNormalize.pkl
python -m underwaterStationKeeping.trainRl --algo sac --env-config configs/rl_stage3_full.json --total-timesteps 300000 --run-name rl_curr_stage3_full --load-model runs/rl_curr_stage2_recover/finalModel.zip --load-vecnormalize runs/rl_curr_stage2_recover/vecNormalize.pkl
```

Evaluate the final curriculum policy:

```powershell
python -m underwaterStationKeeping.evalRl --algo sac --env-config configs/rl_stage3_full.json --model runs/rl_curr_stage3_full/finalModel.zip --vecnormalize runs/rl_curr_stage3_full/vecNormalize.pkl --episodes 10 --output results/rl_curr_eval.csv
```
