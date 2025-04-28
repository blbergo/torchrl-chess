
from typing import Tuple
import torch
from pathlib import Path
from datetime import datetime
from random_word import RandomWords
import os
from torchrl.objectives import PPOLoss, KLPENPPOLoss, ClipPPOLoss
from torchrl.objectives.value import GAE
from torch.optim import Adam
from tensordict.nn import TensorDictModule
from torchrl.collectors import SyncDataCollector
from torchrl.modules import ValueOperator, ProbabilisticActor, ActorCriticWrapper
from actor import ChessActor
from critic import ChessCritic
from torchrl.modules import MaskedCategorical
import torch
from torchrl.envs import ChessEnv
from pathlib import Path
import numpy as np

class Model():
    def __init__(self, value_loss_coef=0.5, entropy_coef=0.2, gamma=0.99, lmda=0.95, lr=1e-4):
        self.log_file_dirs = ["./runs/latest"]
        self.log_files = []
        
        self.env = ChessEnv(include_san=True, include_legal_moves=True)
        n_obs = self.env.observation_spec["legal_moves"].shape[0]
    
        n_actions = self.env.action_spec.n
        
        print(f"observation shape: {n_obs}, action space: {n_actions}")

        actor_net = ChessActor(n_obs, n_actions)

        actor_mod = TensorDictModule(
            module=actor_net,
            in_keys=["legal_moves"],
            out_keys=["logits"],
        )

        self.actor = ProbabilisticActor(
            module=actor_mod,
            in_keys={                               
                "logits": "logits",                # map logits → logits
                "mask": "action_mask",             # map mask   → action_mask
            },
            out_keys=["action"],
            distribution_class=MaskedCategorical,   # applies the mask under the hood
            return_log_prob=True,                  # returns log_prob
        )

        critic_net = ChessCritic(n_obs)

        self.critic = ValueOperator(
            module=critic_net,
            in_keys=["legal_moves"],
            out_keys=["state_value"],
        )

        self.policy = ActorCriticWrapper(
            self.actor,
            self.critic,
        )
        
        self.loss_fn = PPOLoss(
            actor_network=self.actor,
            critic_network=self.critic,
            entropy_coef=entropy_coef,
            critic_coef=value_loss_coef,
        )

        self.advantage = GAE(
            value_network=self.critic,
            gamma=gamma,
            lmbda=lmda,
        )
        
        self.optimizer = Adam(
            self.loss_fn.parameters(),
            lr=lr,
        )
        
    def _init_logs(self):
        for dir in self.log_file_dirs:
            path = Path(dir)
            
            for file in path.parent.glob("*"):
                if file.is_file():
                    file.unlink()
            
            path.mkdir(parents=True, exist_ok=True)
            self.log_files.append(open(f"{dir}/logs.csv", "w"))
            self.log_files[-1].write("episode,loss,loss_type\n")
    
    def _log(self, data):
        for log_file in self.log_files:
            log_file.write(data)
            
    def _flush_logs(self):
        for log_file in self.log_files:
            log_file.flush()
            os.fsync(log_file.fileno())
            
    def _close_logs(self):
        for log_file in self.log_files:
            log_file.close()
    
    def fit(self, episodes=100, max_steps_per_episode=1000, save_rate=0.1):
        save_interval = max(np.floor(episodes * save_rate), 1)
        save_date = datetime.now().strftime("%Y%m%d%H%M%S")

        random_word = RandomWords()
        model_salt = f"{random_word.get_random_word()}_{random_word.get_random_word()}"
        salted_model_name = f"{save_date}_{model_salt}"

        try:
            # Create directories for saving models and logs
            run_path = Path(f"./runs/previous/{salted_model_name}")
            run_path.mkdir(parents=True, exist_ok=True)
            
            # Clear latest directory
            latest_path = Path("./runs/latest")
            for file in latest_path.glob("*"):
                if file.is_file():
                    file.unlink()
                    
        except Exception as e:
            print(f"Error creating directories: {e}")
            exit(1)
            
        self.log_file_dirs.append(f"./runs/previous/{salted_model_name}")
        self._init_logs()
        
        # Initialize the data collector
        collector = SyncDataCollector(
            self.env,
            self.policy,
            total_frames=episodes * max_steps_per_episode,
            frames_per_batch=max_steps_per_episode,
        )
        
        print("Starting training...")
        
        for episode, data in enumerate(collector):
            self.advantage(data.exclude("state_value"))
                    
            total_loss = 0
            loss = self.loss_fn(data)
            for k, v in loss.items():
                if "loss" in k:
                    total_loss += v
                    self._log(f"{episode},{v.item()},{k}\n")
            self._flush_logs()
            
            print(f"Loss: {total_loss.item()}")
            total_loss.backward()
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if episode % save_interval == 0:
                # Save the model
                torch.save({
                    "actor_state_dict": self.actor.state_dict(),
                    "critic_state_dict": self.critic.state_dict(),
                }, f"./runs/previous/{salted_model_name}/model_{episode}.pth")
                
                # Save the latest model
                torch.save({
                    "actor_state_dict": self.actor.state_dict(),
                    "critic_state_dict": self.critic.state_dict(),
                }, f"./runs/latest/model.pth")
                
        self._close_logs()
            

    def load(self, path: str):
        if not os.path.exists(path):
            print(f"Model file {path} does not exist.")
            return

        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        print(f"Model loaded from {path}")