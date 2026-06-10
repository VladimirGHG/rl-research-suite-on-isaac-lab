# Gemini copy pasted code, TODO: (might discard this completely and use clean RL direct implementation with customization, even if this code works i gotta understand it )verification, customization and understanding

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------
# PPO Agent Architecture
# ---------------------------------------------------------
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOAgent(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        # Flatten observation space (e.g., 3x84x84 image -> 21168)
        self.flat_dim = np.prod(obs_shape)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.flat_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(self.flat_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        x_flat = x.reshape(x.shape[0], -1)
        return self.critic(x_flat)

    def get_action_and_value(self, x, action=None, deterministic=False):
        x_flat = x.reshape(x.shape[0], -1)
        action_mean = self.actor_mean(x_flat)
        
        if deterministic:
            # Used purely for the eval() function
            return action_mean, None, None, self.critic(x_flat)
            
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x_flat)

# ---------------------------------------------------------
# Proximal Policy Optimization Trainer Wrapper
# ---------------------------------------------------------
class PPOTrainer:
    def __init__(self, env, algo_cfg, wandb_run=None, resume_path=None):
        # 1. Map Inputs (Matching SAC Signature)
        self.env = env
        self.cfg = algo_cfg
        self.wandb = wandb_run
        self.resume_path = resume_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 2. Extract Dimensions from Vectorized Isaac Lab Env
        self.obs_shape = self.env.observation_space.shape
        self.action_dim = self.env.action_space.shape[0]

        # 3. Initialize Agent & Optimizer
        self.agent = PPOAgent(self.obs_shape, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.cfg.lr, eps=1e-5)

        # 4. Storage / Logging Directories (Matching SAC format)
        # Using getattr to safely fall back if your config names differ slightly
        domain = getattr(self.cfg, "domain", "isaac")
        task = getattr(self.cfg, "task", "manipulation")
        seed = getattr(self.cfg, "seed", 1)
        
        path = f"./log/{domain}_{task}"
        self.exp_dir = os.path.join(path, f"seed_{seed}")
        self.model_dir = os.path.join(self.exp_dir, "models")
        self.tensorboard_dir = os.path.join(self.exp_dir, "tensorboard")

        # 5. Checkpointing Logic (Matching SAC format)
        if getattr(self.cfg, "resume", False) or resume_path is not None:
            load_path = resume_path if resume_path else os.path.join(self.model_dir, "backup.ckpt")
            checkpoint = torch.load(load_path, map_location=self.device)
            
            self.start_iteration = checkpoint['iteration'] + 1
            self.global_step = checkpoint['global_step']
            self.agent.load_state_dict(checkpoint['agent'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Done loading PPO checkpoint from {load_path}...")
        else:
            self.start_iteration = 1
            self.global_step = 0
            # Create directories if starting fresh
            os.makedirs(self.exp_dir, exist_ok=True)
            os.makedirs(self.tensorboard_dir, exist_ok=True)
            os.makedirs(self.model_dir, exist_ok=True)

        # 6. Allocate PPO Rollout Buffers (Vectorized)
        if getattr(self.cfg, "mode", "train") == "train":
            self.num_steps = getattr(self.cfg, "num_steps", 128)
            self.num_envs = getattr(self.cfg, "num_envs", 128)
            
            self.obs_buffer = torch.zeros((self.num_steps, self.num_envs) + self.obs_shape).to(self.device)
            self.action_buffer = torch.zeros((self.num_steps, self.num_envs, self.action_dim)).to(self.device)
            self.logprob_buffer = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
            self.reward_buffer = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
            self.done_buffer = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
            self.value_buffer = torch.zeros((self.num_steps, self.num_envs)).to(self.device)

            # Kickstart the env loop
            raw_obs, _ = self.env.reset()
            self.next_obs = raw_obs["policy"].to(self.device)
            self.next_done = torch.zeros(self.num_envs).to(self.device)

            # Enter training loop
            self.train()
            
        elif getattr(self.cfg, "mode", "train") == "eval":
            print("Entering Evaluation Mode...")
            self.eval(episodes=getattr(self.cfg, "episodes", 10))

    # ---------------------------------------------------------
    # Core Methods
    # ---------------------------------------------------------
    def save_checkpoint(self, name):
        checkpoint = {'agent': self.agent.state_dict()}
        torch.save(checkpoint, os.path.join(self.model_dir, name))

    def save_backup(self, iteration):
        checkpoint = {
            'iteration': iteration,
            'global_step': self.global_step,
            'agent': self.agent.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(self.model_dir, "backup.ckpt"))

    def collect_rollouts(self):
        """Vectorized rollout collection for On-Policy updates."""
        for step in range(self.num_steps):
            self.global_step += self.num_envs
            self.obs_buffer[step] = self.next_obs
            self.done_buffer[step] = self.next_done
            
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(self.next_obs)
                self.value_buffer[step] = value.flatten()
                
            self.action_buffer[step] = action
            self.logprob_buffer[step] = logprob
            
            # Step vectorized envs
            raw_next_obs, reward, terminated, truncated, infos = self.env.step(action)
            done = terminated | truncated
            
            self.reward_buffer[step] = reward.to(self.device).view(-1)
            self.next_obs = raw_next_obs["policy"].to(self.device)
            self.next_done = done.to(self.device).float()

    def update(self):
        """GAE Calculation and Mini-batch Optimization."""
        # 1. GAE
        with torch.no_grad():
            next_value = self.agent.get_value(self.next_obs).flatten()
            advantages = torch.zeros_like(self.reward_buffer).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = 1.0 - self.next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.done_buffer[t + 1]
                    nextvalues = self.value_buffer[t + 1]
                delta = self.reward_buffer[t] + self.cfg.gamma * nextvalues * nextnonterminal - self.value_buffer[t]
                advantages[t] = lastgaelam = delta + self.cfg.gamma * getattr(self.cfg, "gae_lambda", 0.95) * nextnonterminal * lastgaelam
            returns = advantages + self.value_buffer

        # 2. Flatten Arrays
        b_obs = self.obs_buffer.reshape((-1,) + self.obs_shape)
        b_logprobs = self.logprob_buffer.reshape(-1)
        b_actions = self.action_buffer.reshape((-1, self.action_dim))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.value_buffer.reshape(-1)

        # 3. Optimize
        batch_size = self.num_steps * self.num_envs
        mini_batch_size = getattr(self.cfg, "batch_size", 256)
        update_epochs = getattr(self.cfg, "update_epochs", 4)
        clip_coef = getattr(self.cfg, "clip_coef", 0.2)
        
        b_inds = np.arange(batch_size)
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - getattr(self.cfg, "ent_coef", 0.01) * entropy_loss + v_loss * getattr(self.cfg, "vf_coef", 0.5)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), getattr(self.cfg, "max_grad_norm", 0.5))
                self.optimizer.step()

        return pg_loss.item(), v_loss.item(), entropy_loss.item()

    def train(self):
        writer = SummaryWriter(log_dir=self.tensorboard_dir)
        total_iterations = getattr(self.cfg, "episodes", 1000) # Re-using his episodes flag as total iterations

        print("=========================================")
        print(f"Starting PPO Training for {total_iterations} Iterations")
        print("=========================================")

        for iteration in range(self.start_iteration, total_iterations + 1):
            
            # Phase 1: Collect Data
            self.collect_rollouts()
            
            # Phase 2: Update Network
            pg_loss, v_loss, ent_loss = self.update()

            # Logging
            writer.add_scalar("losses/policy_loss", pg_loss, self.global_step)
            writer.add_scalar("losses/value_loss", v_loss, self.global_step)
            writer.add_scalar("losses/entropy", ent_loss, self.global_step)
            
            # Evaluation / Checkpointing matched to SAC frequency logic
            eval_every = getattr(self.cfg, "eval_every", 50)
            if iteration % eval_every == 0 or iteration == total_iterations:
                print(f"[Iter {iteration}] Running Evaluation...")
                eval_ep_r = self.eval(episodes=getattr(self.cfg, "eval_over", 10))
                writer.add_scalar('eval_ep_r', np.mean(eval_ep_r), self.global_step)
                self.save_checkpoint(f"{iteration}.ckpt")
                
            if iteration % 250 == 0 or iteration == total_iterations:
                self.save_backup(iteration)

    def eval(self, episodes, render=False, save_video=False):
        """Vectorized evaluation utilizing deterministic actions."""
        ep_r_list = []
        raw_obs, _ = self.env.reset()
        obs = raw_obs["policy"].to(self.device)
        
        # Track rewards for the vector batch
        current_ep_rewards = torch.zeros(self.num_envs).to(self.device)
        completed_episodes = 0
        
        while completed_episodes < episodes:
            with torch.no_grad():
                # Pass deterministic=True to bypass log_std noise
                action, _, _, _ = self.agent.get_action_and_value(obs, deterministic=True)
                
            raw_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated | truncated
            
            current_ep_rewards += reward.to(self.device).flatten()
            obs = raw_obs["policy"].to(self.device)
            
            # Log rewards for environments that just finished
            for i, is_done in enumerate(done):
                if is_done:
                    ep_r_list.append(current_ep_rewards[i].item())
                    current_ep_rewards[i] = 0.0
                    completed_episodes += 1
                    if completed_episodes >= episodes:
                        break
                        
        avg_reward = np.mean(ep_r_list)
        if getattr(self.cfg, "mode", "train") == "eval":
            print(f"Average Evaluation Return over {episodes} episodes: {avg_reward:.2f}")
            
        return ep_r_list