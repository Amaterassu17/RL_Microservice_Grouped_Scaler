import os
import psutil
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from gymnasium import *
from datetime import datetime
import csv
import wandb

class ActorNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_):
        input_ = torch.FloatTensor(input_)
        output = self.relu(self.fc1(input_))
        output = self.relu(self.fc2(output))
        output = self.fc3(output)
        output = self.softmax(output)
        return output


class CriticNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, input_):
        input_ = torch.FloatTensor(input_)
        output = self.relu(self.fc1(input_))
        output = self.relu(self.fc2(output))
        output = self.fc3(output)
        return output





class PPO_AWARE_AGENT:
    def __init__(self,rl_params, observation_space, action_space, associated_agent , device, writer):
        # self.env = env
        self.associated_agent= associated_agent
        self.app_name = rl_params.app_name
        self.timestamp_creation = rl_params.timestamp
        self.state_size = self.get_state_size(observation_space)
        self.action_size = action_space.n
        self.writer = writer


        self.actor = ActorNetwork(self.state_size, 64, self.action_size)
        self.critic = CriticNetwork(self.state_size, 64)
        
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=rl_params.learning_rate)
        self.cov = torch.diag(torch.ones(self.action_size) * 0.5)
        self.skip_update = False
        self.num_same_parameter_actor = 0
        self.num_same_parameter_critic = 0
        self.parameter_actor = None
        self.parameter_critic = None
        self.recent_rewards = []
        
        self.gamma = rl_params.gamma
        self.k_epochs = rl_params.k_epochs
        self.eps_clip = rl_params.eps_clip
        self.device = device
        self.ent_coef = rl_params.ent_coef
        self.vf_coef = rl_params.vf_coef
        self.max_grad_norm = rl_params.max_grad_norm
        self.critic_loss_discount = rl_params.critic_loss_discount
        self.minibatch_size = rl_params.minibatch_size
        # self.max_num_same_parameters = rl_params.max_num_same_parameters
        self.max_num_rewards_to_check = rl_params.max_num_rewards_to_check
        self.lam = rl_params.gae_lam

    def get_state_size(self, observation_space):
        size = 0
        for space in observation_space:
            if isinstance(space, spaces.Box):
                size += space.shape[0]
            elif isinstance(space, spaces.Discrete):
                size += 1
        return size

    def flatten_state(self, state):
        flat_state = []
        for s in state:
            # print(s)
            # print(type(s))
            if isinstance(s, np.ndarray) or isinstance(s, list):

                flat_state.extend(s)
            else:
                flat_state.append(float(s))  # Convert discrete values to float
        return np.array(flat_state)

    

    def disable_update(self):
        self.skip_update = True

    def enable_update(self):
        self.skip_update = False

    def select_action(self, state):
        state = torch.FloatTensor(self.flatten_state(state))
        action_probs = self.actor(state)
        
        dist = torch.distributions.Categorical(action_probs)
    
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach() + 1, log_prob.detach()



    def learn(self, states, actions, rewards, log_probs, iteration):
        # Transform states into appropriate tensor format
        new_states = [np.concatenate((np.array(state[0]), [state[1]],
                                      np.array(state[2]), 
                                    #   np.array(state[3])
                                      )) for state in states]
        states = torch.FloatTensor(new_states).to(self.device)
        
        #to actions sum -1 to all fields
        actions = torch.tensor(actions, dtype=torch.long).to(self.device) - 1
        log_probs = torch.FloatTensor(log_probs).to(self.device)

        # Compute values using the critic
        with torch.no_grad():
            values = self.critic(states).squeeze().cpu().numpy()

        # Calculate GAE and normalize advantage
        returns = self.calc_gae(rewards, values, self.gamma, self.lam).to(self.device)

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        values = torch.FloatTensor(values).to(self.device)
        advantage = returns - values
        
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # Compute explained variance
        variance_returns = np.var(returns.cpu().numpy())
        variance_residuals = np.var((advantage).cpu().numpy())
        ev = -(1 - (variance_residuals / variance_returns))

        
        returns = returns.squeeze()
        values = values.squeeze()
        advantage = advantage.squeeze()


        total_actor_loss = 0
        total_critic_loss = 0
        total_loss = 0
        total_entropy = 0
        total_value_loss = 0
        total_policy_gradient_loss = 0
        total_approx_kl = 0
        total_clip_fraction = 0

        for epoch in range(self.k_epochs):
            batch_size = states.size(0)
            indices = np.arange(batch_size)
            np.random.shuffle(indices)

            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mini_batch_indices = indices[start:end]

                sampled_states = states[mini_batch_indices]
                sampled_actions = actions[mini_batch_indices]
                sampled_log_probs = log_probs[mini_batch_indices]
                sampled_returns = returns[mini_batch_indices]
                sampled_advantage = advantage[mini_batch_indices]

                values = self.critic(sampled_states).squeeze()
                action_probs = self.actor(sampled_states).squeeze()
                dist = torch.distributions.Categorical(action_probs)
                log_probs_new = dist.log_prob(sampled_actions)
                entropy = dist.entropy().mean()

                ratios = (log_probs_new - sampled_log_probs).exp()
                surrogate1 = ratios * sampled_advantage
                surrogate2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * sampled_advantage
                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                critic_loss = (sampled_returns - values).pow(2).mean()

                loss = actor_loss + self.critic_loss_discount * critic_loss - self.ent_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), self.max_grad_norm)
                self.optimizer.step()

                # Compute additional metrics
                value_loss = critic_loss.item()
                policy_gradient_loss = actor_loss.item()
                approx_kl = (sampled_log_probs - log_probs_new).mean().item()
                clip_fraction = (ratios > 1 + self.eps_clip).float().mean().item() + (ratios < 1 - self.eps_clip).float().mean().item()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_loss += loss.item()
                total_entropy += entropy.item()
                total_value_loss += value_loss
                total_policy_gradient_loss += policy_gradient_loss
                total_approx_kl += approx_kl
                total_clip_fraction += clip_fraction

        

        # Average metrics over the number of updates
        num_updates = (self.k_epochs * (batch_size // self.minibatch_size))
        avg_actor_loss = total_actor_loss / num_updates
        avg_critic_loss = total_critic_loss / num_updates
        avg_total_loss = total_loss / num_updates
        avg_entropy = total_entropy / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_policy_gradient_loss = total_policy_gradient_loss / num_updates
        avg_approx_kl = total_approx_kl / num_updates
        avg_clip_fraction = total_clip_fraction / num_updates

        # Learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        
        

        # Log data after all epochs
        action_distribution = np.bincount(actions.cpu().numpy(), minlength=self.action_size)

        self.log_tensorboard(iteration, avg_actor_loss, avg_critic_loss, avg_total_loss, avg_entropy, 
                            action_distribution, rewards, avg_value_loss, avg_policy_gradient_loss, 
                            avg_approx_kl, avg_clip_fraction, current_lr)

        dict_result = {
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'total_loss': avg_total_loss,
            'entropy': avg_entropy,
            'value_loss': avg_value_loss,
            'policy_gradient_loss': avg_policy_gradient_loss,
            'approx_kl': avg_approx_kl,
            'clip_fraction': avg_clip_fraction,
            'learning_rate': current_lr,
            'action_distribution': action_distribution,
            'explained_variance': ev
        }
        return dict_result

    def calc_gae(self, rewards, values, gamma=0.99, lam=0.95):
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        values = torch.tensor(values, dtype=torch.float32, device=self.device)
        
        returns = torch.zeros_like(rewards)
        gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + gamma * next_value - values[i]
            gae = delta + gamma * lam * gae
            returns[i] = gae + values[i]
            
        return returns

    def explained_variance(returns, predicted_values):
        variance_returns = np.var(returns.cpu().numpy())
        variance_residuals = np.var((returns - predicted_values).cpu().numpy())
        return 1 - (variance_residuals / variance_returns)


    # load all model parameters from a saved checkpoint
    def load_checkpoint(self, checkpoint_file_path):
        if os.path.isfile(checkpoint_file_path):
            print('Loading checkpoint...')
            checkpoint = torch.load(checkpoint_file_path)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('Checkpoint successfully loaded!')
        else:
            raise OSError('Checkpoint not found!')

    # save all model parameters to a checkpoint
    def save_checkpoint(self, episode_num, timestamp_creation, model):
        # convert timestamp_creation to timestamp
        times = timestamp_creation
        checkpoint_dir = CHECKPOINT_DIR + self.app_name + '/' +str(times) + "_" + model + "/agent" + str(self.associated_agent)
        checkpoint_name = checkpoint_dir + '/ep' + str(episode_num) + '.pth.tar' 
        os.makedirs(os.path.dirname(checkpoint_name), exist_ok=True)
        print(f"Saving checkpoint to {checkpoint_name}")
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        torch.save(checkpoint, checkpoint_name)

    # record trajectories
    def save_trajectories(self, iteration, states, actions, rewards, log_probs):
        timestamp = int(self.timestamp_creation)
        directory = os.path.join('./logs/', self.app_name, str(timestamp) + "_ppo_aware", "agent" + str(self.associated_agent))
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = os.path.join(directory, "ppo_trajectories.csv")
        file_exists = os.path.isfile(file_path)

        
        
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # Write header only if file does not exist
            if not file_exists:
                writer.writerow(['Iteration', 'Avg_CPU', 'Avg_MEM', "Std_dev_CPU", "Std_dev_MEM", 'Num_replicas', 'Scale_Num_replica_Chosen', 'Reward', 'Log_Prob'])
            
            count = 0

            for _ ,state in enumerate(states):
                # Assuming state is a tuple like (array1, array2, scalar)
                state_flattened = [item.item() if isinstance(item, np.generic) else item for substate in state for item in (substate if isinstance(substate, np.ndarray) else [substate])]
                if len(actions) != count:
                    action = actions[count].item()
                    reward = rewards[count].item() if isinstance(rewards, np.generic) else rewards[count]
                    log_prob = log_probs[count].item()
                else:
                    action = None
                    reward = None
                    log_prob = None
                # Debug prints
                # print(f"   --> State: {state_flattened}")
                # print(f"   --> Action: {action}")
                # print(f"     --> Reward: {reward}")
                # print(f"       --> Log prob: {log_prob}")
                
                writer.writerow([iteration] + state_flattened + [action, reward, log_prob])
                count += 1
                
                
    def visualize_tensorboard(self):
        
        print(f"Agent {self.associated_agent}",  self.actor)
        print(f"Agent {self.associated_agent}",  self.critic)

    def log_tensorboard(self, iteration, actor_loss, critic_loss, total_loss, entropy, action_distribution, 
                        rewards, value_loss, policy_gradient_loss, approx_kl, clip_fraction, learning_rate):
        
        # TensorBoard logging
        self.writer.add_scalar(f'Agent_{self.associated_agent}/Loss/Actor Loss', actor_loss, iteration)
        self.writer.add_scalar(f'Agent_{self.associated_agent}/Loss/Critic Loss', critic_loss, iteration)
        self.writer.add_scalar(f'Agent_{self.associated_agent}/Loss/Total Loss', total_loss, iteration)
        self.writer.add_scalar(f'Agent_{self.associated_agent}/Entropy', entropy, iteration)
        self.writer.add_scalar(f'Agent_{self.associated_agent}/Loss/Value Loss', value_loss, iteration)
        self.writer.add_scalar(f'Agent_{self.associated_agent}/Policy Gradient Loss', policy_gradient_loss, iteration)
        self.writer.add_scalar(f'Agent_{self.associated_agent}/Approx KL', approx_kl, iteration)
        self.writer.add_scalar(f'Agent_{self.associated_agent}/Clip Fraction', clip_fraction, iteration)
        self.writer.add_scalar(f'Agent_{self.associated_agent}/Learning Rate', learning_rate, iteration)

        # Log mean and standard deviation of rewards for the current iteration
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        self.writer.add_scalar(f'Agent_{self.associated_agent}/Reward/Mean', mean_reward, iteration)
        self.writer.add_scalar(f'Agent_{self.associated_agent}/Reward/Std', std_reward, iteration)
        
        # Log action distribution
        for action, count in enumerate(action_distribution):
            self.writer.add_scalar(f'Agent_{self.associated_agent}/Action/Action_{action}', count, iteration)

        # Log histograms of network parameters
        for name, param in self.actor.named_parameters():
            self.writer.add_histogram(f'Agent_{self.associated_agent}/Actor/{name}', param, iteration)
        for name, param in self.critic.named_parameters():
            self.writer.add_histogram(f'Agent_{self.associated_agent}/Critic/{name}', param, iteration)

        