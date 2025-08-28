# Sonic the Hedgehog PPO Reinforcement Learning System

A production-ready implementation of Proximal Policy Optimization (PPO) for training an agent to play Sonic the Hedgehog Genesis. This system includes comprehensive training, evaluation, video recording, and checkpointing capabilities.

## Features

- 🎮 **Complete PPO Implementation** - Optimized for Sonic with proven hyperparameters
- 📊 **Comprehensive Evaluation** - Performance metrics, transfer learning tests, and action analysis
- 🎬 **Video Recording** - Automatic gameplay recording during training and evaluation
- 💾 **Robust Checkpointing** - Model versioning with automatic cleanup
- 📈 **Real-time Monitoring** - TensorBoard and optional Weights & Biases integration
- 🚀 **Production Ready** - Inference optimization and deployment configurations

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd sonic-ppo-rl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. ROM Setup

#### Legal ROM Acquisition

The recommended legal approach is purchasing the SEGA Mega Drive & Genesis Classics on Steam:

1. Purchase from Steam (~$30 for full collection, often $1 per game on sale)
2. Find ROMs in Steam installation directory:
   ```
   Windows: C:\Program Files (x86)\Steam\steamapps\common\Sega Classics\uncompressed ROMs\
   Linux: ~/.steam/steam/steamapps/common/Sega Classics/uncompressed ROMs/
   Mac: ~/Library/Application Support/Steam/steamapps/common/Sega Classics/uncompressed ROMs/
   ```

3. Import ROMs to OpenAI Retro:
   ```bash
   python -m retro.import /path/to/roms/
   ```

4. Verify installation:
   ```bash
   python -c "import retro; print(retro.data.list_games())"
   ```
   You should see "SonicTheHedgehog-Genesis" in the list.

### 3. Training

#### Basic Training

```bash
# Start training with default settings (20M timesteps)
python train.py --experiment-id my_first_run

# Quick test run (100k timesteps)
python train.py --experiment-id test_run --total-timesteps 100000 --n-envs 4

# Full training with video recording
python train.py --experiment-id full_training \
    --total-timesteps 20000000 \
    --n-envs 8 \
    --record-video \
    --use-wandb
```

#### Advanced Training Options

```bash
# Train on specific level
python train.py --experiment-id marble_zone \
    --state MarbleZone.Act1 \
    --total-timesteps 10000000

# Distributed training with more environments
python train.py --experiment-id distributed \
    --n-envs 16 \
    --start-method spawn \
    --device cuda

# Custom hyperparameters (modify config.py or use command line)
python train.py --experiment-id custom_hp \
    --checkpoint-freq 50000 \
    --eval-freq 25000 \
    --video-freq 50000
```

### 4. Evaluation

```bash
# Evaluate a trained model
python eval.py experiments/my_first_run/final_model.zip \
    --n-episodes 50 \
    --deterministic

# Comprehensive evaluation with transfer learning tests
python eval.py experiments/my_first_run/best_model/best_model.zip \
    --test-transfer \
    --analyze-actions \
    --benchmark-speed \
    --create-plots

# Record gameplay videos
python eval.py experiments/my_first_run/final_model.zip \
    --record-video \
    --n-videos 5 \
    --video-dir gameplay_videos
```

## Configuration

### Key Hyperparameters (config.py)

The most critical hyperparameters based on successful implementations:

```python
PPO_CONFIG = {
    "learning_rate": 2.5e-4,      # With linear annealing
    "n_steps": 2048,              # Steps per environment per update  
    "batch_size": 64,             # Larger than default for stability
    "n_epochs": 10,               # PPO epochs per update
    "ent_coef": 0.04,             # CRITICAL: High entropy prevents "always right" problem
    "gamma": 0.99,                # Discount factor
    "gae_lambda": 0.95,           # GAE lambda
}
```

### Using Configuration Presets

```python
from config import get_config

# Quick testing
config = get_config(preset="quick_test")

# Standard training
config = get_config(preset="standard")

# Competition-grade training
config = get_config(preset="competition")

# Custom configuration
config = get_config(
    preset="standard",
    total_timesteps=30_000_000,
    n_envs=16,
    use_curriculum=True
)
```

## Project Structure

```
sonic-ppo-rl/
├── train.py           # Main training script with PPO implementation
├── eval.py            # Comprehensive evaluation and testing
├── config.py          # Centralized configuration and hyperparameters
├── requirements.txt   # Python dependencies
├── README.md          # This file
├── experiments/       # Training outputs (created automatically)
│   └── <experiment_id>/
│       ├── checkpoints/     # Model checkpoints
│       ├── videos/          # Training videos
│       ├── logs/            # Training logs
│       ├── eval_logs/       # Evaluation metrics
│       ├── best_model/      # Best performing model
│       └── config.json      # Experiment configuration
└── evaluation_results/  # Evaluation outputs
    ├── plots/              # Performance visualizations
    ├── gameplay_videos/    # Recorded gameplay
    └── evaluation_report.json
```

## Training Tips

### 1. Avoiding Common Pitfalls

**The "Always Press Right" Problem**
- **Solution**: Set entropy coefficient to 0.04+ (4x higher than Atari defaults)
- **Monitor**: Check that entropy doesn't decay to zero
- **Early detection**: Episode rewards plateau at 1000-2000

**Reward Scaling Issues**
- **Critical**: Always use 0.01 reward scaling for Sonic
- **Symptoms**: Exploding gradients, unstable training

**Environment Wrapper Order**
- **Correct order**: Base → Action → FrameSkip → Warp → Stack → Scale → Monitor
- **Never change**: This order is critical for proper training

### 2. Performance Expectations

- **2-5M timesteps**: Basic movement competency
- **10-15M timesteps**: Consistent level completion
- **20M+ timesteps**: Human-level performance
- **Training time**: 8-16 hours on RTX 3070

### 3. Hardware Requirements

**Minimum**:
- GPU: GTX 1060 6GB
- CPU: Intel i5-8400 / AMD Ryzen 5 2600
- RAM: 16GB
- Storage: 20GB free space

**Recommended**:
- GPU: RTX 3070 or better
- CPU: Intel i7-10700 / AMD Ryzen 7 3700X
- RAM: 32GB
- Storage: 50GB SSD

## Monitoring Training

### TensorBoard

```bash
# Launch TensorBoard
tensorboard --logdir experiments/<experiment_id>/logs

# View at http://localhost:6006
```

### Weights & Biases (Optional)

```bash
# Login to W&B
wandb login

# Train with W&B logging
python train.py --use-wandb --experiment-id wandb_run
```

## Troubleshooting

### Common Issues

1. **"ROM not found" error**
   ```bash
   # List available games
   python -c "import retro; print(retro.data.list_games())"
   
   # Re-import ROMs
   python -m retro.import /path/to/roms/
   ```

2. **CUDA out of memory**
   ```bash
   # Reduce number of parallel environments
   python train.py --n-envs 4
   
   # Or force CPU usage
   python train.py --device cpu
   ```

3. **Training instability**
   - Check reward scaling (should be 0.01)
   - Increase entropy coefficient
   - Reduce learning rate
   - Check for NaN values in logs

4. **Slow training**
   - Use more parallel environments (`--n-envs 16`)
   - Ensure GPU is being used (`--device cuda`)
   - Disable video recording during training
   - Use SSD for checkpoint storage

## Advanced Features

### Curriculum Learning

Modify `config.py` to enable curriculum learning:

```python
CURRICULUM_CONFIG = {
    "use_curriculum": True,
    "curriculum_type": "sequential",
    "curriculum_stages": [
        {"levels": ["GreenHillZone.Act1"], "timesteps": 5_000_000},
        {"levels": ["GreenHillZone.Act1", "GreenHillZone.Act2"], "timesteps": 5_000_000},
        {"levels": ["GreenHillZone.Act1", "GreenHillZone.Act2", "GreenHillZone.Act3"], "timesteps": 10_000_000},
    ]
}
```

### Multi-Level Training

Train on multiple levels simultaneously:

```python
# Modify train.py to use multiple states
states = ["GreenHillZone.Act1", "GreenHillZone.Act2", "MarbleZone.Act1"]
# Create environments with different states for each parallel env
```

### Custom Reward Shaping

Implement trajectory-based rewards or curiosity-driven exploration by modifying the reward wrapper in `train.py`.

## Performance Benchmarks

Expected performance after full training (20M timesteps):

| Level | Completion Rate | Mean Reward | Max X Position |
|-------|----------------|-------------|----------------|
| Green Hill Zone Act 1 | 95%+ | 8000+ | 9000+ |
| Green Hill Zone Act 2 | 85%+ | 7000+ | 8500+ |
| Green Hill Zone Act 3 | 75%+ | 6500+ | 8000+ |
| Marble Zone Act 1 (Transfer) | 40%+ | 3000+ | 4000+ |

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is provided for educational purposes. Ensure you own legal copies of any game ROMs used.

## Acknowledgments

- OpenAI for the Retro Contest and baseline implementations
- Stable-Baselines3 team for the robust PPO implementation
- Contest participants whose approaches informed these optimizations

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{sonic_ppo_rl,
  title={Sonic the Hedgehog PPO Reinforcement Learning System},
  year={2024},
  url={https://github.com/yourusername/sonic-ppo-rl}
}
```

## Contact

For questions or support, please open an issue on GitHub.
