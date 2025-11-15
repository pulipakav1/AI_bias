## Bias Project

This project evaluates large language models for bias, toxicity, stereotype reinforcement, and sentiment variance across demographic axes. It ingests benchmark datasets (such as Crows-Pairs), runs configurable model queries, and aggregates results into metrics, plots, and summary tables.

### Repo Map
- `metrics.py`, `fairness_metrics.py`: scoring utilities
- `data_loader.py`, `model_interface.py`: dataset/model abstractions
- `visualization.py`: chart creation helpers

