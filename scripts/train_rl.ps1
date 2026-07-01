$ErrorActionPreference = "Stop"

python -m underwaterStationKeeping.trainRl `
  --algo sac `
  --total-timesteps 300000 `
  --run-name rl_box_station_sac

python -m underwaterStationKeeping.evalRl `
  --algo sac `
  --model runs/rl_box_station_sac/finalModel.zip `
  --vecnormalize runs/rl_box_station_sac/vecNormalize.pkl `
  --episodes 5 `
  --output results/rl_eval.csv
