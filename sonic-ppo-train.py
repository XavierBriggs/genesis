#!/usr/bin/env python3
"""
Sonic the Hedgehog Genesis PPO Training Script
Main training pipeline with checkpointing, video recording, and monitoring
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

import retro
import gym
from gym import spaces
from gym.wrappers import FrameStack

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Optional integrations
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


# ============= Environment Wrappers =============

class SonicDiscretizer(gym.ActionWrapper):
    """
    Discretize Sonic action space to essential 8 actions for efficient exploration
    """
    def __init__(self, env):
        super().__init__(env)
        # Essential Sonic actions based on contest analysis
        self._actions = [
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # NOOP
            np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),  # LEFT
            np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),  # RIGHT
            np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]),  # LEFT + DOWN
            np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]),  # RIGHT + DOWN
            np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),  # DOWN
            np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),  # DOWN + B (spin dash)
            np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # B (jump)
        ]
        self.action_space = spaces.Discrete(len(self._actions))
        
    def action(self, a):
        return self._actions[a]


class AllowBacktracking(gym.Wrapper):
    """
    Use max(X) progress instead of delta(X) for better exploration
    """
    def __init__(self, env):
        super().__init__(env)
        self._max_x = 0
        self._cur_x = 0
        
    def reset(self, **kwargs):
        self._max_x = 0
        self._cur_x = 0
        return self.env.reset(**kwargs)
        
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info


class RewardScaler(gym.RewardWrapper):
    """
    Scale rewards by 0.01 for PPO stability (critical for Sonic)
    """
    def __init__(self, env, scale=0.01):
        super().__init__(env)
        self.scale = scale
        
    def reward(self, reward):
        return reward * self.scale


class WarpFrame(gym.ObservationWrapper):
    """
    Warp frames to 84x84 as done in Atari preprocessing
    Keep RGB channels for Sonic (better than grayscale)
    """
    def __init__(self, env, width=84, height=84, grayscale=False):
        super().__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        
        if self.grayscale:
            num_colors = 1
        else:
            num_colors = 3
            
        new_space = spaces.Box(
            low=0, high=255,
            shape=(self.height, self.width, num_colors),
            dtype=np.uint8
        )
        
        original_space = self.observation_space
        self.observation_space = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3
        
    def observation(self, frame):
        import cv2
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        
        if self.grayscale:
            frame = np.expand_dims(frame, -1)
            
        return frame


class StochasticFrameSkip(gym.Wrapper):
    """
    Stochastic frame skipping to make environment more stochastic
    """
    def __init__(self, env, n, stickprob=0.25):
        super().__init__(env)
        self.n = n
        self.stickprob = stickprob
        self.last_action = 0
        
    def step(self, action):
        if np.random.uniform() < self.stickprob:
            action = self.last_action
        self.last_action = action
        
        total_reward = 0.0
        done = False
        
        for i in range(self.n):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
                
        return obs, total_reward, done, info


# ============= Custom CNN Architecture =============

class SonicCNN(BaseFeaturesExtractor):
    """
    Custom CNN for Sonic based on Nature CNN with RGB optimization
    """
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        # We assume CHW format from stable-baselines3
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate output dimension
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, *observation_space.shape)).shape[1]
            
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
        
    def forward(self, observations):
        return self.linear(self.cnn(observations))


# ============= Custom Callbacks =============

class VideoRecorderCallback(BaseCallback):
    """
    Record videos at specified intervals during training
    """
    def __init__(self, eval_env, render_freq=100_000, n_eval_episodes=1, 
                 video_folder="videos", name_prefix="sonic_video"):
        super().__init__()
        self.eval_env = eval_env
        self.render_freq = render_freq
        self.n_eval_episodes = n_eval_episodes
        self.video_folder = video_folder
        self.name_prefix = name_prefix
        self.video_counter = 0
        
        os.makedirs(video_folder, exist_ok=True)
        
    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            self.record_video()
        return True
        
    def record_video(self):
        from gym.wrappers import RecordVideo
        
        video_name = f"{self.name_prefix}_{self.video_counter}_step_{self.num_timesteps}"
        self.video_counter += 1
        
        # Wrap environment with video recorder
        video_env = RecordVideo(
            self.eval_env,
            video_folder=self.video_folder,
            name_prefix=video_name,
            episode_trigger=lambda x: x < self.n_eval_episodes
        )
        
        # Run evaluation episodes
        for episode in range(self.n_eval_episodes):
            obs = video_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = video_env.step(action)
                episode_reward += reward
                
            print(f"Video recorded - Episode reward: {episode_reward:.2f}")
            
        video_env.close()


class TrainingMetricsCallback(BaseCallback):
    """
    Custom callback to track and log training metrics
    """
    def __init__(self, log_freq=1000):
        super().__init__()
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log metrics
        if self.n_calls % self.log_freq == 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-100:])
                mean_length = np.mean(self.episode_lengths[-100:])
                
                self.logger.record("rollout/mean_episode_reward", mean_reward)
                self.logger.record("rollout/mean_episode_length", mean_length)
                
        return True
        
    def _on_rollout_end(self) -> None:
        # Collect episode statistics
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])


# ============= Environment Creation =============

def make_sonic_env(game="SonicTheHedgehog-Genesis", state="GreenHillZone.Act1", 
                   record_video=False, video_folder="videos"):
    """
    Create and wrap a Sonic environment with all necessary wrappers
    """
    def _init():
        env = retro.make(game=game, state=state, scenario="contest")
        
        # Apply wrappers in correct order
        env = SonicDiscretizer(env)
        env = StochasticFrameSkip(env, n=4, stickprob=0.25)
        env = WarpFrame(env, width=84, height=84, grayscale=False)  # RGB for Sonic
        env = AllowBacktracking(env)
        env = RewardScaler(env, scale=0.01)
        
        # Add monitoring
        env = Monitor(env)
        
        # Optional video recording
        if record_video:
            from gym.wrappers import RecordVideo
            env = RecordVideo(
                env,
                video_folder=video_folder,
                episode_trigger=lambda x: x % 100 == 0
            )
            
        return env
        
    return _init


def create_envs(n_envs=8, game="SonicTheHedgehog-Genesis", 
                state="GreenHillZone.Act1", start_method='spawn'):
    """
    Create vectorized environments for parallel training
    """
    env_fns = [make_sonic_env(game=game, state=state) for _ in range(n_envs)]
    
    if n_envs > 1:
        envs = SubprocVecEnv(env_fns, start_method=start_method)
    else:
        envs = DummyVecEnv(env_fns)
        
    # Frame stacking (4 frames)
    envs = VecFrameStack(envs, n_stack=4)
    
    return envs


# ============= Training Configuration =============

def get_ppo_config():
    """
    Get optimized PPO hyperparameters for Sonic
    Based on contest winners and empirical testing
    """
    return {
        "learning_rate": 2.5e-4,           # With linear annealing
        "n_steps": 2048,                   # Steps per env per update
        "batch_size": 64,                  # Minibatch size
        "n_epochs": 10,                    # PPO epochs
        "gamma": 0.99,                     # Discount factor
        "gae_lambda": 0.95,                # GAE lambda
        "clip_range": 0.2,                 # PPO clipping
        "clip_range_vf": None,             # Value function clipping
        "ent_coef": 0.04,                  # CRITICAL: High entropy for exploration
        "vf_coef": 0.5,                    # Value function coefficient
        "max_grad_norm": 0.5,              # Gradient clipping
        "target_kl": None,                 # Early stopping of updates
        "policy_kwargs": {
            "features_extractor_class": SonicCNN,
            "features_extractor_kwargs": {"features_dim": 512},
            "net_arch": [512, 512],        # Two hidden layers
            "activation_fn": nn.ReLU,
        }
    }


# ============= Main Training Function =============

def train(args):
    """
    Main training function
    """
    # Setup experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"sonic_ppo_{args.experiment_id}_{timestamp}"
    
    # Create directories
    save_dir = Path(args.save_dir) / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    video_dir = save_dir / "videos"
    video_dir.mkdir(exist_ok=True)
    log_dir = save_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config.update(get_ppo_config())
    
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
        
    print(f"Starting experiment: {experiment_name}")
    print(f"Save directory: {save_dir}")
    
    # Initialize wandb if available
    if WANDB_AVAILABLE and args.use_wandb:
        run = wandb.init(
            project="sonic-ppo",
            name=experiment_name,
            config=config,
            sync_tensorboard=True,
        )
        
    # Create environments
    print("Creating training environments...")
    train_envs = create_envs(
        n_envs=args.n_envs,
        game=args.game,
        state=args.state,
        start_method=args.start_method
    )
    
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_sonic_env(game=args.game, state=args.state)])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    
    # Configure logger
    logger = configure(str(log_dir), ["stdout", "tensorboard"])
    
    # Create model
    print("Initializing PPO model...")
    ppo_config = get_ppo_config()
    
    model = PPO(
        "CnnPolicy",
        train_envs,
        **ppo_config,
        tensorboard_log=str(log_dir),
        verbose=1,
        seed=args.seed,
        device=args.device,
    )
    
    model.set_logger(logger)
    
    # Setup callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(save_dir / "checkpoints"),
        name_prefix="sonic_ppo_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / "best_model"),
        log_path=str(save_dir / "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)
    
    # Video recording callback
    if args.record_video:
        video_callback = VideoRecorderCallback(
            eval_env,
            render_freq=args.video_freq,
            n_eval_episodes=1,
            video_folder=str(video_dir),
            name_prefix="training_progress"
        )
        callbacks.append(video_callback)
        
    # Training metrics callback
    metrics_callback = TrainingMetricsCallback(log_freq=1000)
    callbacks.append(metrics_callback)
    
    # Wandb callback
    if WANDB_AVAILABLE and args.use_wandb:
        wandb_callback = WandbCallback(
            model_save_path=str(save_dir / "wandb_models"),
            verbose=2,
        )
        callbacks.append(wandb_callback)
        
    # Combine callbacks
    callback = CallbackList(callbacks)
    
    # Train model
    print(f"\nStarting training for {args.total_timesteps} timesteps...")
    print(f"Using {args.n_envs} parallel environments")
    print(f"Checkpoint frequency: {args.checkpoint_freq} timesteps")
    print(f"Evaluation frequency: {args.eval_freq} timesteps")
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        
    # Save final model
    print("Saving final model...")
    model.save(save_dir / "final_model")
    
    # Clean up
    train_envs.close()
    eval_env.close()
    
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.finish()
        
    print(f"Training complete! Results saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train PPO on Sonic the Hedgehog")
    
    # Experiment settings
    parser.add_argument("--experiment-id", type=str, default="default",
                      help="Experiment identifier")
    parser.add_argument("--save-dir", type=str, default="./experiments",
                      help="Directory to save models and logs")
    
    # Environment settings
    parser.add_argument("--game", type=str, default="SonicTheHedgehog-Genesis",
                      help="Sonic game to use")
    parser.add_argument("--state", type=str, default="GreenHillZone.Act1",
                      help="Level/state to train on")
    parser.add_argument("--n-envs", type=int, default=8,
                      help="Number of parallel environments")
    parser.add_argument("--start-method", type=str, default="spawn",
                      choices=["spawn", "fork", "forkserver"],
                      help="Multiprocessing start method")
    
    # Training settings
    parser.add_argument("--total-timesteps", type=int, default=20_000_000,
                      help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                      choices=["auto", "cuda", "cpu"],
                      help="Device to use for training")
    
    # Evaluation and logging
    parser.add_argument("--eval-freq", type=int, default=50_000,
                      help="Evaluation frequency in timesteps")
    parser.add_argument("--n-eval-episodes", type=int, default=10,
                      help="Number of episodes for evaluation")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000,
                      help="Checkpoint save frequency")
    parser.add_argument("--record-video", action="store_true",
                      help="Record videos during training")
    parser.add_argument("--video-freq", type=int, default=100_000,
                      help="Video recording frequency")
    parser.add_argument("--use-wandb", action="store_true",
                      help="Use Weights & Biases for logging")
    
    args = parser.parse_args()
    
    # Check if retro is properly set up
    try:
        import retro
        retro.data.list_games()
    except Exception as e:
        print(f"Error: OpenAI Retro not properly configured: {e}")
        print("Please install retro and import your Sonic ROM:")
        print("  pip install gym-retro")
        print("  python -m retro.import /path/to/your/roms/")
        sys.exit(1)
        
    train(args)


if __name__ == "__main__":
    main()
