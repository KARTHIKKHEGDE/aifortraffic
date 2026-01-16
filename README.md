# Multi-Agent Reinforcement Learning for Adaptive Traffic Signal Control

## Bangalore Real Intersections with Emergency Vehicle Priority and Weather-Aware Optimization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![SUMO 1.15+](https://img.shields.io/badge/SUMO-1.15+-green.svg)](https://sumo.dlr.de/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

This project implements a **production-grade Multi-Agent Reinforcement Learning (MARL)** system for adaptive traffic signal control operating on **real Bangalore city intersections**. The system uses actual OpenStreetMap data and SUMO microscopic traffic simulation.

### Target Intersections

1. **Silk Board Junction** (Primary - Most congested intersection in India)
2. **Tin Factory Junction**
3. **Hebbal Flyover Junction**
4. **Marathahalli - Outer Ring Road Junction**

### Key Features

- ✅ Real Bangalore road network from OpenStreetMap
- ✅ Multi-Agent RL controlling 4 interconnected junctions
- ✅ Emergency vehicle (ambulance) priority with green corridor
- ✅ Weather-aware signal timing (rain adaptation)
- ✅ Configurable queue length system (baseline → realistic → calibrated)
- ✅ Comprehensive evaluation against fixed-time and actuated baselines

## 📊 Performance Targets

| Metric                       | Fixed-Time  | RL Target    | Improvement |
| ---------------------------- | ----------- | ------------ | ----------- |
| Avg Waiting Time             | 180s        | <120s        | 33%+        |
| Queue Length                 | 45 vehicles | <30 vehicles | 33%+        |
| Emergency Clearance          | 90s         | <30s         | 67%+        |
| Rain Performance Degradation | 40%         | <15%         | 62%+        |

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    BANGALORE RL TRAFFIC CONTROL                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Silk Board  │  │ Tin Factory │  │   Hebbal    │  ...        │
│  │   Agent     │  │   Agent     │  │   Agent     │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          ▼                                      │
│              ┌──────────────────────┐                           │
│              │   PPO Policy Network │                           │
│              └──────────────────────┘                           │
│                          │                                      │
│         ┌────────────────┼────────────────┐                     │
│         ▼                ▼                ▼                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Emergency  │  │   Weather   │  │   Queue     │             │
│  │  Priority   │  │   Model     │  │   Config    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                          │                                      │
│                          ▼                                      │
│              ┌──────────────────────┐                           │
│              │   SUMO Simulator     │                           │
│              │   (TraCI Interface)  │                           │
│              └──────────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- SUMO 1.15+ (with TraCI)
- CUDA-capable GPU (recommended for training)

### Installation

```bash
# 1. Clone/navigate to project
cd aifortraffic

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install SUMO (if not installed)
# Windows: Download from https://sumo.dlr.de/docs/Downloads.php
# Set SUMO_HOME environment variable

# 5. Acquire and prepare maps
python scripts/01_download_osm.py
python scripts/02_convert_to_sumo.py

# 6. Generate traffic demand
python scripts/03_generate_routes.py

# 7. Train RL agents
python scripts/04_train_curriculum.py --agent dqn

# 8. Evaluate
python scripts/05_evaluate.py --agent-type dqn --model-path runs/<experiment>/final_model_dqn.pt --report

# 9. Visualize results
python scripts/06_visualize.py --results runs/<experiment>/training_results.json

# 10. Run demo
python scripts/07_demo.py --agent-type dqn --model-path runs/<experiment>/final_model_dqn.pt
```

## 📁 Project Structure

```
aifortraffic/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── configs/                    # Configuration files
│   ├── env_config.yaml         # Environment settings
│   ├── training_config.yaml    # Training hyperparameters
│   └── junctions.yaml          # Bangalore junction coordinates
├── data/                       # Raw data
│   └── osm/                    # OpenStreetMap files
├── maps/                       # SUMO network files
├── routes/                     # Traffic demand files
├── src/                        # Source code
│   ├── __init__.py
│   ├── environment/            # RL environment
│   │   ├── __init__.py
│   │   ├── traffic_env.py      # Gymnasium environment
│   │   ├── sumo_connector.py   # TraCI interface
│   │   └── queue_config.py     # Queue length modes
│   ├── agents/                 # RL algorithms
│   │   ├── __init__.py
│   │   ├── qlearning.py        # Tabular Q-Learning
│   │   ├── dqn_agent.py        # Deep Q-Network
│   │   └── ppo_agent.py        # PPO Multi-Agent
│   ├── emergency/              # Emergency priority
│   │   ├── __init__.py
│   │   └── priority_handler.py # Ambulance detection & override
│   ├── weather/                # Weather modeling
│   │   ├── __init__.py
│   │   └── weather_model.py    # Bangalore rain patterns
│   ├── evaluation/             # Metrics & evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py          # Traffic metrics
│   │   ├── baselines.py        # Fixed-time & actuated
│   │   └── analyzer.py         # Results analysis
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── config.py           # Configuration management
│       ├── logger.py           # Logging utilities
│       └── sumo_utils.py       # SUMO helpers
├── scripts/                    # Executable scripts
│   ├── 01_download_osm.py      # Download OSM data
│   ├── 02_convert_to_sumo.py   # Convert to SUMO network
│   ├── 03_generate_routes.py   # Generate traffic routes
│   ├── 04_train_curriculum.py  # Curriculum learning
│   ├── 05_evaluate.py          # Evaluate agents
│   ├── 06_visualize.py         # Generate plots
│   └── 07_demo.py              # Interactive demo
├── runs/                       # Training experiments
└── tests/                      # Unit tests
```

## 🎮 Usage

### Training

```bash
# Train with Q-Learning (fast, for testing)
python scripts/04_train_curriculum.py --agent qlearning

# Train with DQN (recommended)
python scripts/04_train_curriculum.py --agent dqn --device cuda

# Train with PPO Multi-Agent (best performance)
python scripts/04_train_curriculum.py --agent ppo --device cuda

# Train specific curriculum stage
python scripts/04_train_curriculum.py --agent dqn --stage 2

# Resume from checkpoint
python scripts/04_train_curriculum.py --agent dqn --checkpoint runs/.../checkpoint.pt
```

### Evaluation

```bash
# Evaluate trained agent vs baselines
python scripts/05_evaluate.py \
    --agent-type dqn \
    --model-path runs/<experiment>/final_model_dqn.pt \
    --episodes 20 \
    --report

# Evaluate specific junction
python scripts/05_evaluate.py \
    --agent-type dqn \
    --model-path runs/<experiment>/final_model_dqn.pt \
    --junction silk_board
```

### Visualization

```bash
# Generate all plots
python scripts/06_visualize.py --results evaluation_results.json --plot-type all

# Generate specific plot types
python scripts/06_visualize.py --results evaluation_results.json --plot-type training
python scripts/06_visualize.py --results evaluation_results.json --plot-type comparison
python scripts/06_visualize.py --results evaluation_results.json --plot-type radar
```

### Demo

```bash
# Run interactive demo with SUMO-GUI
python scripts/07_demo.py \
    --agent-type dqn \
    --model-path runs/<experiment>/final_model_dqn.pt \
    --junction silk_board \
    --render

# Compare baselines only
python scripts/07_demo.py --baseline-only --junction silk_board
```

## ⚙️ Configuration

### Environment Config (`configs/env_config.yaml`)

```yaml
# Key settings
simulation:
  step_length: 1.0
  max_steps: 3600
  gui: false

state:
  queue_mode: "realistic_bangalore" # baseline, realistic_bangalore, calibrated

reward:
  waiting_time_weight: -0.4
  queue_length_weight: -0.3
  emergency_weight: -0.5
  throughput_weight: 0.2

emergency:
  enabled: true
  priority_time: 30
  detection_range: 200

weather:
  enabled: true
  rain_probability: 0.15
```

### Training Config (`configs/training_config.yaml`)

```yaml
# Curriculum stages
curriculum:
  stages:
    - name: basic
      episodes: 100
      weather_enabled: false
      emergency_enabled: false
    - name: weather
      episodes: 150
      weather_enabled: true
    - name: emergency
      episodes: 200
      emergency_enabled: true
    - name: full
      episodes: 300
      weather_enabled: true
      emergency_enabled: true
      multi_junction: true

# Agent hyperparameters
dqn:
  learning_rate: 0.001
  gamma: 0.95
  epsilon_decay: 0.995
  buffer_size: 100000
  batch_size: 64
  target_update_freq: 500
  hidden_layers: [256, 256, 128]
  double: true
  dueling: false

ppo:
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  clip_range: 0.2
```

## 📊 Queue Length Modes

The system supports three queue length simulation modes:

| Mode                  | Description                        | Queue Multiplier |
| --------------------- | ---------------------------------- | ---------------- |
| `baseline`            | Standard SUMO values               | 1.0x             |
| `realistic_bangalore` | Bangalore congestion patterns      | 1.5-2.8x         |
| `calibrated`          | Sensor-calibrated (when available) | Custom           |

Junction-specific multipliers:

- **Silk Board**: 2.8x (India's most congested)
- **Marathahalli**: 2.2x
- **Tin Factory**: 2.0x
- **Hebbal**: 1.8x

## 📖 Documentation

- [Architecture Design](docs/architecture.md)
- [Training Guide](docs/training_guide.md)
- [Evaluation Protocol](docs/evaluation_protocol.md)
- [API Reference](docs/api_reference.md)

## 🔬 Research Contributions

1. **First RL system on actual Bangalore congestion hotspots**
2. **Configurable queue length simulation** (baseline → realistic → calibrated)
3. **Emergency priority integrated into MARL framework**
4. **Weather-aware reward function with safety constraints**
5. **Transferable methodology for Indian cities**

## 📝 Citation

```bibtex
@article{bangalore_rl_traffic_2024,
  title={Multi-Agent Reinforcement Learning for Emergency-Aware Traffic Signal Control on Real Urban Networks: A Bangalore Case Study},
  author={Your Name},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2024}
}
```

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.
#   a i f o r t r a f f i c  
 