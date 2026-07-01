$ErrorActionPreference = "Stop"

python -m underwaterStationKeeping.runScenario --scenario mild --output results/mild_run.csv
python -m underwaterStationKeeping.runScenario --scenario strong --output results/strong_run.csv
python -m underwaterStationKeeping.runScenario --scenario reversing --output results/reversing_run.csv

python -m underwaterStationKeeping.plotResults --input results/mild_run.csv --output results/mild_run.png
python -m underwaterStationKeeping.plotResults --input results/strong_run.csv --output results/strong_run.png
python -m underwaterStationKeeping.plotResults --input results/reversing_run.csv --output results/reversing_run.png
