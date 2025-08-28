"""
Sonic the Hedgehog PPO Configuration
Centralized configuration for hyperparameters and training settings
"""

import torch.nn as nn
from pathlib import Path


# ============= Environment Configuration =============

ENV_CONFIG = {
    # Game settings
    "game": "SonicTheHedgehog-Genesis",
    "scenario": "contest",
    
    # Training levels (for curriculum learning)
    "training_levels": [
        "GreenHillZone.Act1",
        "GreenHillZone.Act2", 
        "GreenHillZone.Act3",
    ],
    
    # Test levels (for transfer evaluation)
    "test_levels": [
        "MarbleZone.Act1",
        "MarbleZone.Act2",
        "SpringYardZone.Act1",
        "LabyrinthZone.Act1",
        "StarLightZone.Act1",
        "ScrapBrainZone.Act1",
    ],
    
    # Parallel environments
    "n_envs": 8,
    "start_method": "spawn",  # 'spawn', 'fork', or 'forkserver'
    
    # Frame processing
    "frame_skip": 4,
    "frame_stack": 4,
    "frame_width": 84,
    "frame_height": 84,
    "grayscale": False,  # RGB works better for Sonic
    "stick_prob": 0.25,  # Stochastic frame skip probability
    
    # Reward settings
    "reward_scale": 0.01,  # Critical for PPO stability
    "allow_backtracking": True,  # Use max(X) instead of delta(X)
}


# ============= PPO Hyperparameters =============

PPO_CONFIG = {
    # Core PPO parameters (optimized for Sonic)
    "learning_rate": 2.5e-4,           # With linear annealing
    "n_steps": 2048,                   # Steps per env per update
    "batch_size": 64,                  # Minibatch size 
    "n_epochs": 10,                    # PPO epochs per update
    "gamma": 0.99,                     # Discount factor
    "gae_lambda": 0.95,                # GAE lambda
    "clip_range": 0.2,                 # PPO clipping parameter
    "clip_range_vf": None,             # Value function clipping (None = no clipping)
    
    # CRITICAL: High entropy coefficient prevents "always press right" problem
    "ent_coef": 0.04,                  # Entropy coefficient (4x higher than Atari)
    "vf_coef": 0.5,                    # Value function coefficient
    "max_grad_norm": 0.5,              # Gradient clipping
    "target_kl": None,                 # KL divergence threshold (None = no early stopping)
    
    # Learning rate schedule
    "use_linear_schedule": True,       # Linear decay of learning rate
    "lr_schedule_power": 1.0,          # Power for polynomial schedule (1.0 = linear)
}


# ============= Network Architecture =============

NETWORK_CONFIG = {
    # CNN feature extractor
    "features_dim": 512,               # Size of CNN output
    "use_custom_cnn": True,            # Use custom CNN architecture
    
    # Nature CNN layers (optimized for 84x84 RGB input)
    "cnn_layers": [
        # (out_channels, kernel_size, stride)
        (32, 8, 4),
        (64, 4, 2),
        (64, 3, 1),
    ],
    
    # Policy/Value heads
    "net_arch": [512, 512],            # Hidden layers for policy and value networks
    "activation_fn": nn.ReLU,          # Activation function
    "use_sde": False,                  # State-Dependent Exploration
    "log_std_init": 0.0,               # Initial log std for continuous actions
    "ortho_init": True,                # Orthogonal initialization
}


# ============= Training Configuration =============

TRAINING_CONFIG = {
    # Training duration
    "total_timesteps": 20_000_000,     # 20M steps for full training
    "warmup_timesteps": 100_000,       # Optional warmup period
    
    # Evaluation settings
    "eval_freq": 50_000,               # Evaluation frequency in timesteps
    "n_eval_episodes": 10,             # Episodes per evaluation
    "eval_deterministic": True,        # Use deterministic policy for eval
    
    # Checkpointing
    "checkpoint_freq": 100_000,        # Save model every N timesteps
    "keep_checkpoints": 5,             # Number of checkpoints to keep
    "save_best_model": True,           # Save best model based on eval
    
    # Video recording
    "record_video": True,              # Enable video recording
    "video_freq": 100_000,             # Record video every N timesteps
    "video_episodes": 1,               # Episodes to record per video
    "video_length": 500,               # Max frames per video
    
    # Logging
    "log_interval": 10,                # Log every N updates
    "tensorboard": True,               # Enable TensorBoard logging
    "wandb": False,                    # Enable Weights & Biases logging
    "wandb_project": "sonic-ppo",      # W&B project name
    "wandb_entity": None,              # W&B entity/team name
    
    # Random seed
    "seed": 42,                        # Random seed for reproducibility
    
    # Device settings
    "device": "auto",                  # 'auto', 'cuda', or 'cpu'
    "force_cpu": False,                # Force CPU even if GPU available
}


# ============= Reward Shaping Configuration =============

REWARD_CONFIG = {
    # Reward components and weights
    "progress_weight": 0.75,           # X-position progress
    "time_bonus_weight": 0.15,         # Speed completion bonus
    "exploration_weight": 0.05,        # Novel state exploration
    "ring_collection_weight": 0.05,    # Ring collection bonus
    
    # Reward shaping parameters
    "use_trajectory_reward": False,    # Use human demonstration trajectory
    "trajectory_weight": 0.1,          # Weight for trajectory following
    "use_curiosity_reward": False,     # Intrinsic curiosity module
    "curiosity_weight": 0.05,          # Weight for curiosity bonus
    
    # Penalty settings
    "death_penalty": -10,              # Penalty for losing a life
    "time_penalty": -0.01,             # Per-step time penalty
    "backward_penalty": -0.5,          # Penalty for moving backward
}


# ============= Curriculum Learning Configuration =============

CURRICULUM_CONFIG = {
    "use_curriculum": False,            # Enable curriculum learning
    "curriculum_type": "sequential",   # 'sequential', 'random', or 'adaptive'
    
    # Sequential curriculum stages
    "curriculum_stages": [
        {
            "levels": ["GreenHillZone.Act1"],
            "timesteps": 5_000_000,
        },
        {
            "levels": ["GreenHillZone.Act1", "GreenHillZone.Act2"],
            "timesteps": 5_000_000,
        },
        {
            "levels": [
                "GreenHillZone.Act1", 
                "GreenHillZone.Act2",
                "GreenHillZone.Act3"
            ],
            "timesteps": 10_000_000,
        },
    ],
    
    # Adaptive curriculum settings
    "adaptive_threshold": 0.8,         # Success rate to advance
    "adaptive_window": 100,            # Episodes to measure success
}


# ============= Advanced Training Features =============

ADVANCED_CONFIG = {
    # Distributed training
    "use_distributed": False,           # Enable distributed training
    "n_workers": 1,                    # Number of distributed workers
    
    # Mixed precision training
    "use_mixed_precision": False,      # Enable mixed precision (FP16)
    "loss_scale": "dynamic",           # Loss scaling for FP16
    
    # Memory optimization
    "use_recurrent_policy": False,     # Use LSTM/GRU policy
    "lstm_hidden_size": 256,           # LSTM hidden size
    "n_lstm_layers": 1,                # Number of LSTM layers
    
    # Exploration strategies
    "exploration_type": "entropy",     # 'entropy', 'epsilon', or 'noisy_net'
    "epsilon_start": 1.0,              # Initial epsilon for epsilon-greedy
    "epsilon_end": 0.01,               # Final epsilon
    "epsilon_decay": 0.995,            # Epsilon decay rate
    
    # Auxiliary tasks
    "use_auxiliary_tasks": False,      # Enable auxiliary tasks
    "auxiliary_tasks": [               # List of auxiliary tasks
        "next_frame_prediction",
        "reward_prediction",
    ],
}


# ============= Production Deployment Configuration =============

DEPLOYMENT_CONFIG = {
    # Model optimization
    "optimize_for_inference": True,    # Optimize model for deployment
    "use_quantization": False,         # Quantize model (INT8)
    "use_torchscript": False,          # Convert to TorchScript
    "use_onnx": False,                 # Export to ONNX format
    
    # Performance monitoring
    "monitor_inference": True,          # Track inference performance
    "inference_batch_size": 1,         # Batch size for inference
    "max_inference_time_ms": 50,       # Max allowed inference time
    
    # Fallback behavior
    "use_fallback_policy": True,       # Enable fallback for errors
    "fallback_action": "random",       # 'random' or 'noop'
    
    # API settings
    "api_enabled": False,              # Enable REST API
    "api_host": "0.0.0.0",            # API host
    "api_port": 8000,                 # API port
    "api_max_concurrent": 10,         # Max concurrent requests
}


# ============= Paths and Directories =============

PATH_CONFIG = {
    "base_dir": Path("./experiments"),
    "rom_dir": Path("./roms"),
    "checkpoint_dir": Path("./checkpoints"),
    "video_dir": Path("./videos"),
    "log_dir": Path("./logs"),
    "tensorboard_dir": Path("./tensorboard"),
    "model_dir": Path("./models"),
    "evaluation_dir": Path("./evaluation"),
}


# ============= Preset Configurations =============

PRESETS = {
    "quick_test": {
        # Fast training for testing setup
        "total_timesteps": 100_000,
        "n_envs": 4,
        "checkpoint_freq": 10_000,
        "eval_freq": 10_000,
        "n_eval_episodes": 3,
    },
    
    "standard": {
        # Standard training configuration
        "total_timesteps": 10_000_000,
        "n_envs": 8,
        "checkpoint_freq": 100_000,
        "eval_freq": 50_000,
    },
    
    "competition": {
        # Competition-grade training
        "total_timesteps": 50_000_000,
        "n_envs": 16,
        "use_curriculum": True,
        "use_trajectory_reward": True,
        "checkpoint_freq": 250_000,
        "eval_freq": 100_000,
        "n_eval_episodes": 20,
    },
    
    "debug": {
        # Debug configuration with verbose output
        "total_timesteps": 10_000,
        "n_envs": 1,
        "log_interval": 1,
        "checkpoint_freq": 1_000,
        "eval_freq": 1_000,
        "n_eval_episodes": 1,
        "video_freq": 1_000,
    },
}


def get_config(preset="standard", **overrides):
    """
    Get configuration with optional preset and overrides
    
    Args:
        preset: Name of preset configuration
        **overrides: Key-value pairs to override config
        
    Returns:
        Complete configuration dictionary
    """
    config = {
        **ENV_CONFIG,
        **PPO_CONFIG,
        **NETWORK_CONFIG,
        **TRAINING_CONFIG,
        **REWARD_CONFIG,
        **CURRICULUM_CONFIG,
        **ADVANCED_CONFIG,
        **DEPLOYMENT_CONFIG,
        **PATH_CONFIG,
    }
    
    # Apply preset if specified
    if preset in PRESETS:
        preset_config = PRESETS[preset]
        config.update(preset_config)
        
    # Apply manual overrides
    config.update(overrides)
    
    # Create directories
    for key, path in config.items():
        if key.endswith("_dir") and isinstance(path, Path):
            path.mkdir(parents=True, exist_ok=True)
            
    return config


def save_config(config, filepath):
    """Save configuration to JSON file"""
    import json
    
    # Convert Path objects to strings
    config_json = {}
    for key, value in config.items():
        if isinstance(value, Path):
            config_json[key] = str(value)
        elif hasattr(value, '__name__'):  # Functions/classes
            config_json[key] = value.__name__
        else:
            config_json[key] = value
            
    with open(filepath, 'w') as f:
        json.dump(config_json, f, indent=2)
        

def load_config(filepath):
    """Load configuration from JSON file"""
    import json
    
    with open(filepath, 'r') as f:
        config = json.load(f)
        
    # Convert string paths back to Path objects
    for key, value in config.items():
        if key.endswith("_dir") and isinstance(value, str):
            config[key] = Path(value)
            
    return config
