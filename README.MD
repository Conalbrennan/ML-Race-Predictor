# F1 Race Predictor –  2025

A machine learning-based simulation tool that forecasts the finishing positions of drivers in a Formula 1 Grand Prix. The current model is configured to predict the results of the 2025 Austrian Grand Prix, using historical race data, current qualifying performance, and driver-specific trends from earlier in the season.

This project leverages the FastF1 API to access real-world F1 telemetry and results, and combines that with manually supplied qualifying data from 2025. The result is a flexible prediction pipeline that can adapt to new races, drivers, and seasons with minimal manual effort.

---

## Features

- Predicts finishing order for a given F1 race (default: Austria 2025)
- Combines:
  - 2024 race session data (lap time, compound, DNF)
  - 2025 qualifying times
  - Past 4-race trends (qualifying consistency, finish average, DNF rate)
- Automatically adapts if used on a different target race/round
- Trained using a Gradient Boosting Regressor

## Requirements

Install dependencies via pip:

```bash
pip install -r requirements.txt

Required libraries:

fastf1

pandas

scikit-learn


