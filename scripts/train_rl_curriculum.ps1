$ErrorActionPreference = "Stop"

python -m underwaterStationKeeping.trainRl `
  --algo sac `
  --env-config configs/rl_stage1_hold.json `
  --total-timesteps 150000 `
  --run-name rl_curr_stage1_hold `
  --ent-coef auto_0.1

python -m underwaterStationKeeping.trainRl `
  --algo sac `
  --env-config configs/rl_stage2_recover.json `
  --total-timesteps 200000 `
  --run-name rl_curr_stage2_recover `
  --load-model runs/rl_curr_stage1_hold/finalModel.zip `
  --load-vecnormalize runs/rl_curr_stage1_hold/vecNormalize.pkl

python -m underwaterStationKeeping.trainRl `
  --algo sac `
  --env-config configs/rl_stage3_full.json `
  --total-timesteps 300000 `
  --run-name rl_curr_stage3_full `
  --load-model runs/rl_curr_stage2_recover/finalModel.zip `
  --load-vecnormalize runs/rl_curr_stage2_recover/vecNormalize.pkl

python -m underwaterStationKeeping.evalRl `
  --algo sac `
  --env-config configs/rl_stage3_full.json `
  --model runs/rl_curr_stage3_full/finalModel.zip `
  --vecnormalize runs/rl_curr_stage3_full/vecNormalize.pkl `
  --episodes 10 `
  --output results/rl_curr_eval.csv
