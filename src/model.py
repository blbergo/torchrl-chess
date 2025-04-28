
from typing import Tuple
import torch
from pathlib import Path
from datetime import datetime
from random_word import RandomWords
import os
from torchrl.objectives import PPOLoss
from torchrl.objectives.value import GAE
from torch.optim import Adam
from tensordict.nn import TensorDictModule
from torchrl.modules import ValueOperator, ProbabilisticActor, ActorCriticWrapper
from actor import ChessActor
from critic import ChessCritic
from torchrl.modules import MaskedCategorical
import torch
from torchrl.envs import ChessEnv
from pathlib import Path

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
            policy_operator=self.actor,
            value_operator=self.critic,
        )
        
        self.loss_fn = PPOLoss(
            actor_network=self.policy.get_policy_operator(),
            critic_network=self.policy.get_value_operator(),
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
        )

        self.advantage = GAE(
            value_network=self.policy.get_value_operator(),
            gamma=gamma,
            lmbda=lmda,
        )

        self.optimizer = Adam(
            list(self.policy.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )
        
    def _init_logs(self):
        for dir in self.log_file_dirs:
            path = Path(dir)
            
            for file in path.parent.glob("*"):
                if file.is_file():
                    file.unlink()
            
            path.mkdir(parents=True, exist_ok=True)
            self.log_files.append(open(f"{dir}/logs.csv", "a"))
            self.log_files[-1].write("episode,loss,loss_type\n")
            
        for log_file in self.log_files:
            log_file.write("episode,loss,loss_type\n")
    
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
    
    def train(self, episodes=100, max_steps_per_episode=1000, save_rate=0.1):
        save_interval = int(episodes * save_rate)
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

        for episode in range(episodes):
            rollout = self.env.rollout(
                policy=self.policy,
                max_steps=max_steps_per_episode,
            )
            
            rollout = rollout.select(
                "next",
                "legal_moves",
                "done",
                "action",
                "action_mask",
                "sample_log_prob",
            )
            self.advantage(rollout)
            # detach sample_log_prob
            rollout["sample_log_prob"].detach_()
            loss = self.loss_fn(rollout)
            
            total_loss = 0
            for k, v in loss.items():
                self._log(f"{episode},{v.item()},{k}\n")
                total_loss += v
            self._flush_logs()
                
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            print(f"Episode {episode + 1}/{episodes} - Loss: {total_loss.item():.4f}")
            
            # Save the model
            if episode % save_interval == 0:
                torch.save(
                    {
                        "actor_state_dict": self.actor.state_dict(),
                        "critic_state_dict": self.critic.state_dict(),
                    },
                    f"./runs/latest/model_{episode + save_interval}.pth",
                )

        self._close_logs()
        # Save the model at the end   
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
            },
            f"./runs/previous/{salted_model_name}/model.pth",
        )
        
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
            },
            f"./runs/latest/model.pth",
        )      
        
    def load(self, path: str):
        if not os.path.exists(path):
            print(f"Model file {path} does not exist.")
            return

        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        print(f"Model loaded from {path}")