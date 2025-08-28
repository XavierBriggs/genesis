#!/usr/bin/env python3
"""
Sonic the Hedgehog Genesis PPO Evaluation Script
Comprehensive evaluation and testing with video recording and metrics analysis
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

import retro
import gym
from gym.wrappers import RecordVideo

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

# Import environment wrappers from train.py
from train import (
    SonicDiscretizer, AllowBacktracking, RewardScaler, 
    WarpFrame, StochasticFrameSkip, make_sonic_env
)


class SonicEvaluator:
    """
    Comprehensive evaluation suite for Sonic PPO models
    """
    
    def __init__(self, model_path, config_path=None, device="auto"):
        """
        Initialize evaluator with trained model
        
        Args:
            model_path: Path to saved model
            config_path: Optional path to training config
            device: Device for model inference
        """
        self.model_path = Path(model_path)
        self.model = PPO.load(str(model_path), device=device)
        
        # Load config if available
        self.config = {}
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                
        # Results storage
        self.results = defaultdict(list)
        
    def create_eval_env(self, game="SonicTheHedgehog-Genesis", 
                        state="GreenHillZone.Act1", record_video=False,
                        video_folder="eval_videos"):
        """
        Create environment for evaluation
        """
        env = DummyVecEnv([make_sonic_env(game=game, state=state)])
        env = VecFrameStack(env, n_stack=4)
        
        if record_video:
            os.makedirs(video_folder, exist_ok=True)
            # Note: Recording with vectorized env requires special handling
            
        return env
        
    def evaluate_single_level(self, game="SonicTheHedgehog-Genesis",
                             state="GreenHillZone.Act1", n_episodes=50,
                             deterministic=True, render=False, 
                             record_video=False, video_dir="eval_videos"):
        """
        Evaluate model performance on a single level
        
        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"\nEvaluating on {game} - {state}")
        print(f"Running {n_episodes} episodes...")
        
        # Create environment
        env = self.create_eval_env(game, state, record_video, video_dir)
        
        # Track metrics
        episode_rewards = []
        episode_lengths = []
        completion_times = []
        max_x_positions = []
        deaths = []
        rings_collected = []
        
        for episode in range(n_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            max_x = 0
            total_rings = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = env.step(action)
                
                episode_reward += reward[0]
                episode_length += 1
                
                # Extract info from vectorized env
                if isinstance(info, list):
                    info = info[0]
                    
                # Track additional metrics if available
                if 'x' in info:
                    max_x = max(max_x, info['x'])
                if 'rings' in info:
                    total_rings = max(total_rings, info.get('rings', 0))
                    
                if render:
                    env.render()
                    
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            max_x_positions.append(max_x)
            rings_collected.append(total_rings)
            
            # Check if episode ended in death
            if info and 'lives' in info:
                deaths.append(info['lives'] < 3)  # Assuming starts with 3 lives
            else:
                deaths.append(False)
                
            # Track completion time if level was completed
            if info and info.get('level_complete', False):
                completion_times.append(episode_length)
                
            if (episode + 1) % 10 == 0:
                print(f"  Episode {episode + 1}/{n_episodes} - "
                      f"Reward: {episode_reward:.2f}, Length: {episode_length}")
                
        env.close()
        
        # Calculate statistics
        results = {
            'level': state,
            'n_episodes': n_episodes,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'mean_x_position': np.mean(max_x_positions),
            'max_x_position': np.max(max_x_positions),
            'completion_rate': len(completion_times) / n_episodes,
            'mean_completion_time': np.mean(completion_times) if completion_times else 0,
            'death_rate': np.mean(deaths),
            'mean_rings': np.mean(rings_collected),
        }
        
        # Store results
        self.results[state] = results
        
        return results
        
    def evaluate_transfer_learning(self, test_levels, n_episodes=20,
                                   deterministic=True):
        """
        Test zero-shot transfer to unseen levels
        
        Args:
            test_levels: List of (game, state) tuples
            n_episodes: Episodes per level
            deterministic: Use deterministic policy
            
        Returns:
            Dictionary of transfer learning results
        """
        print("\n" + "="*50)
        print("TRANSFER LEARNING EVALUATION")
        print("="*50)
        
        transfer_results = {}
        
        for game, state in test_levels:
            results = self.evaluate_single_level(
                game=game,
                state=state,
                n_episodes=n_episodes,
                deterministic=deterministic,
                render=False
            )
            transfer_results[f"{game}_{state}"] = results
            
        return transfer_results
        
    def run_action_analysis(self, game="SonicTheHedgehog-Genesis",
                           state="GreenHillZone.Act1", n_steps=10000):
        """
        Analyze action distribution during gameplay
        
        Returns:
            Action frequency distribution
        """
        print(f"\nAnalyzing action distribution over {n_steps} steps...")
        
        env = self.create_eval_env(game, state)
        
        action_counts = defaultdict(int)
        action_names = [
            "NOOP", "LEFT", "RIGHT", "LEFT+DOWN", 
            "RIGHT+DOWN", "DOWN", "SPIN_DASH", "JUMP"
        ]
        
        obs = env.reset()
        
        for step in range(n_steps):
            action, _ = self.model.predict(obs, deterministic=False)
            action_counts[int(action[0])] += 1
            
            obs, _, done, _ = env.step(action)
            
            if done:
                obs = env.reset()
                
        env.close()
        
        # Convert to percentages
        action_distribution = {}
        for i, name in enumerate(action_names):
            count = action_counts.get(i, 0)
            percentage = (count / n_steps) * 100
            action_distribution[name] = percentage
            
        return action_distribution
        
    def benchmark_inference_speed(self, n_samples=1000):
        """
        Benchmark model inference speed
        
        Returns:
            Dict with timing statistics
        """
        import time
        
        print(f"\nBenchmarking inference speed with {n_samples} samples...")
        
        # Create dummy observations
        obs_shape = (1, 4, 84, 84)  # Batch size 1, 4 stacked frames
        dummy_obs = np.random.rand(*obs_shape).astype(np.float32)
        
        times = []
        
        # Warmup
        for _ in range(10):
            _ = self.model.predict(dummy_obs, deterministic=True)
            
        # Benchmark
        for _ in range(n_samples):
            start = time.perf_counter()
            _ = self.model.predict(dummy_obs, deterministic=True)
            end = time.perf_counter()
            times.append(end - start)
            
        times = np.array(times) * 1000  # Convert to milliseconds
        
        results = {
            'mean_inference_time_ms': np.mean(times),
            'std_inference_time_ms': np.std(times),
            'min_inference_time_ms': np.min(times),
            'max_inference_time_ms': np.max(times),
            'p95_inference_time_ms': np.percentile(times, 95),
            'fps': 1000 / np.mean(times),
        }
        
        return results
        
    def generate_performance_plots(self, save_dir="eval_plots"):
        """
        Generate visualization plots for evaluation results
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if not self.results:
            print("No results to plot. Run evaluation first.")
            return
            
        # Set style
        sns.set_style("whitegrid")
        
        # Plot 1: Reward distribution across levels
        if len(self.results) > 1:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Rewards bar plot
            levels = list(self.results.keys())
            mean_rewards = [self.results[l]['mean_reward'] for l in levels]
            std_rewards = [self.results[l]['std_reward'] for l in levels]
            
            ax = axes[0, 0]
            ax.bar(levels, mean_rewards, yerr=std_rewards, capsize=5)
            ax.set_xlabel('Level')
            ax.set_ylabel('Mean Episode Reward')
            ax.set_title('Performance Across Levels')
            ax.tick_params(axis='x', rotation=45)
            
            # Completion rates
            completion_rates = [self.results[l]['completion_rate'] * 100 for l in levels]
            
            ax = axes[0, 1]
            ax.bar(levels, completion_rates, color='green', alpha=0.7)
            ax.set_xlabel('Level')
            ax.set_ylabel('Completion Rate (%)')
            ax.set_title('Level Completion Success')
            ax.tick_params(axis='x', rotation=45)
            
            # Max X position reached
            max_x_positions = [self.results[l]['max_x_position'] for l in levels]
            
            ax = axes[1, 0]
            ax.bar(levels, max_x_positions, color='orange', alpha=0.7)
            ax.set_xlabel('Level')
            ax.set_ylabel('Max X Position')
            ax.set_title('Furthest Progress in Level')
            ax.tick_params(axis='x', rotation=45)
            
            # Episode lengths
            mean_lengths = [self.results[l]['mean_length'] for l in levels]
            
            ax = axes[1, 1]
            ax.bar(levels, mean_lengths, color='purple', alpha=0.7)
            ax.set_xlabel('Level')
            ax.set_ylabel('Mean Episode Length')
            ax.set_title('Average Episode Duration')
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/performance_summary.png", dpi=150)
            plt.close()
            
        # Plot 2: Single level detailed analysis (if only one level)
        if len(self.results) == 1:
            level_name = list(self.results.keys())[0]
            result = self.results[level_name]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Metrics summary
            metrics = ['mean_reward', 'completion_rate', 'mean_x_position']
            values = [
                result['mean_reward'],
                result['completion_rate'] * 100,
                result['mean_x_position']
            ]
            labels = ['Mean Reward', 'Completion %', 'Mean X Position']
            
            ax = axes[0]
            bars = ax.bar(labels, values, color=['blue', 'green', 'orange'])
            ax.set_title(f'Performance Summary - {level_name}')
            ax.set_ylabel('Value')
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}', ha='center', va='bottom')
                       
            # Death rate pie chart
            ax = axes[1]
            death_rate = result.get('death_rate', 0) * 100
            survival_rate = 100 - death_rate
            
            ax.pie([survival_rate, death_rate], 
                  labels=['Survived', 'Died'],
                  colors=['green', 'red'],
                  autopct='%1.1f%%')
            ax.set_title('Episode Outcomes')
            
            # Statistics box
            ax = axes[2]
            ax.axis('off')
            stats_text = f"""
            Statistics for {level_name}:
            
            Episodes: {result['n_episodes']}
            Mean Reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}
            Max Reward: {result['max_reward']:.2f}
            Completion Rate: {result['completion_rate']*100:.1f}%
            Mean Rings: {result['mean_rings']:.1f}
            """
            ax.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/single_level_analysis.png", dpi=150)
            plt.close()
            
        print(f"Plots saved to {save_dir}/")
        
    def generate_report(self, save_path="evaluation_report.json"):
        """
        Generate comprehensive evaluation report
        """
        report = {
            'model_path': str(self.model_path),
            'evaluation_timestamp': datetime.now().isoformat(),
            'configuration': self.config,
            'level_results': dict(self.results),
        }
        
        # Add action analysis if available
        if hasattr(self, 'action_distribution'):
            report['action_distribution'] = self.action_distribution
            
        # Add inference speed if benchmarked
        if hasattr(self, 'inference_speed'):
            report['inference_speed'] = self.inference_speed
            
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\nEvaluation report saved to {save_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        for level, results in self.results.items():
            print(f"\n{level}:")
            print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"  Completion Rate: {results['completion_rate']*100:.1f}%")
            print(f"  Max X Position: {results['max_x_position']:.0f}")
            print(f"  Death Rate: {results.get('death_rate', 0)*100:.1f}%")
            
    def create_gameplay_video(self, game="SonicTheHedgehog-Genesis",
                             state="GreenHillZone.Act1", n_episodes=3,
                             video_dir="gameplay_videos", deterministic=True):
        """
        Create high-quality gameplay videos for analysis
        """
        print(f"\nRecording {n_episodes} gameplay videos...")
        
        os.makedirs(video_dir, exist_ok=True)
        
        for episode in range(n_episodes):
            # Create base environment
            env = retro.make(game=game, state=state, scenario="contest")
            
            # Apply wrappers
            env = SonicDiscretizer(env)
            env = StochasticFrameSkip(env, n=4, stickprob=0.0 if deterministic else 0.25)
            env = WarpFrame(env, width=84, height=84, grayscale=False)
            env = AllowBacktracking(env)
            env = RewardScaler(env, scale=0.01)
            
            # Add video recording
            video_name = f"{game}_{state}_episode_{episode}"
            env = RecordVideo(
                env,
                video_folder=video_dir,
                name_prefix=video_name,
                episode_trigger=lambda x: True
            )
            
            # Create vectorized env for model compatibility
            vec_env = DummyVecEnv([lambda: env])
            vec_env = VecFrameStack(vec_env, n_stack=4)
            
            # Run episode
            obs = vec_env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = vec_env.step(action)
                total_reward += reward[0]
                steps += 1
                
            vec_env.close()
            
            print(f"  Episode {episode + 1}: Reward={total_reward:.2f}, Steps={steps}")
            
        print(f"Videos saved to {video_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Sonic PPO model")
    
    # Model settings
    parser.add_argument("model_path", type=str,
                      help="Path to trained model file")
    parser.add_argument("--config-path", type=str, default=None,
                      help="Path to training config JSON")
    parser.add_argument("--device", type=str, default="auto",
                      choices=["auto", "cuda", "cpu"],
                      help="Device for inference")
    
    # Evaluation settings
    parser.add_argument("--n-episodes", type=int, default=50,
                      help="Number of evaluation episodes per level")
    parser.add_argument("--deterministic", action="store_true",
                      help="Use deterministic policy")
    
    # Levels to evaluate
    parser.add_argument("--game", type=str, default="SonicTheHedgehog-Genesis",
                      help="Sonic game to evaluate")
    parser.add_argument("--state", type=str, default="GreenHillZone.Act1",
                      help="Primary level to evaluate")
    parser.add_argument("--test-transfer", action="store_true",
                      help="Test transfer learning on multiple levels")
    
    # Analysis options
    parser.add_argument("--analyze-actions", action="store_true",
                      help="Analyze action distribution")
    parser.add_argument("--benchmark-speed", action="store_true",
                      help="Benchmark inference speed")
    parser.add_argument("--create-plots", action="store_true",
                      help="Generate performance plots")
    
    # Video recording
    parser.add_argument("--record-video", action="store_true",
                      help="Record gameplay videos")
    parser.add_argument("--n-videos", type=int, default=3,
                      help="Number of videos to record")
    parser.add_argument("--video-dir", type=str, default="gameplay_videos",
                      help="Directory for video files")
    
    # Output settings
    parser.add_argument("--save-dir", type=str, default="evaluation_results",
                      help="Directory for evaluation outputs")
    
    args = parser.parse_args()
    
    # Create output directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Initialize evaluator
    print(f"Loading model from {args.model_path}...")
    evaluator = SonicEvaluator(args.model_path, args.config_path, args.device)
    
    # Run primary evaluation
    results = evaluator.evaluate_single_level(
        game=args.game,
        state=args.state,
        n_episodes=args.n_episodes,
        deterministic=args.deterministic,
        render=False
    )
    
    # Test transfer learning
    if args.test_transfer:
        transfer_levels = [
            ("SonicTheHedgehog-Genesis", "GreenHillZone.Act2"),
            ("SonicTheHedgehog-Genesis", "GreenHillZone.Act3"),
            ("SonicTheHedgehog-Genesis", "MarbleZone.Act1"),
            ("SonicTheHedgehog-Genesis", "SpringYardZone.Act1"),
        ]
        
        transfer_results = evaluator.evaluate_transfer_learning(
            transfer_levels,
            n_episodes=10,
            deterministic=args.deterministic
        )
        
    # Analyze action distribution
    if args.analyze_actions:
        action_dist = evaluator.run_action_analysis(args.game, args.state)
        evaluator.action_distribution = action_dist
        
        print("\nAction Distribution:")
        for action, percentage in action_dist.items():
            print(f"  {action}: {percentage:.1f}%")
            
    # Benchmark inference speed
    if args.benchmark_speed:
        speed_results = evaluator.benchmark_inference_speed()
        evaluator.inference_speed = speed_results
        
        print("\nInference Speed:")
        print(f"  Mean: {speed_results['mean_inference_time_ms']:.2f}ms")
        print(f"  FPS: {speed_results['fps']:.1f}")
        
    # Generate plots
    if args.create_plots:
        plot_dir = save_dir / "plots"
        evaluator.generate_performance_plots(str(plot_dir))
        
    # Record gameplay videos
    if args.record_video:
        evaluator.create_gameplay_video(
            game=args.game,
            state=args.state,
            n_episodes=args.n_videos,
            video_dir=str(save_dir / args.video_dir),
            deterministic=args.deterministic
        )
        
    # Generate report
    report_path = save_dir / "evaluation_report.json"
    evaluator.generate_report(str(report_path))


if __name__ == "__main__":
    main()