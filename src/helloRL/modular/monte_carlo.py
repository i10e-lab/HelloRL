import torch

from helloRL.modular.rollout_data import ExperienceData
from helloRL.modular.foundation import *
from helloRL.modular.actors import ActorProtocol

class RolloutMethodMonteCarlo(RolloutMethod):
    @property
    def n_envs(self) -> int:
        return 1
    
    def collect_experience_data(
        self, envs: gym.vector.VectorEnv, initial_states: torch.Tensor, actor: ActorProtocol, tracker: SessionTracker
        ) -> ExperienceData:
            states = []
            actions = []
            next_states = []
            rewards = []

            env = envs.envs[0]

            state, _ = env.reset()

            while not tracker.is_session_complete():
                tracker.increment_timestep()

                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # shape: (1, state_space)

                with torch.no_grad():
                    action_t, _ = actor.output(state_t)

                action_np = action_t.squeeze().numpy()
                next_state, reward, terminated, truncated, info = env.step(action_np)

                next_state_t = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)  # shape: (1, state_space)
                
                states.append(state_t)
                actions.append(action_t)
                next_states.append(next_state_t)
                rewards.append(reward)
                
                state = next_state
                
                if terminated or truncated:
                    episode_return = info['episode']['r']
                    episode_length = info['episode']['l']

                    tracker.finish_episode(episode_return, episode_length)
                    break

            states_t = torch.cat(states, dim=0)        # shape: (episode_length, state_space)
            # add an additional dimension for n_envs at dim 0
            states_t = states_t.unsqueeze(0)          # shape: (1 env, episode_length, state_space)
            actions_t = torch.cat(actions, dim=0)      # shape: (episode_length, action_dim)
            actions_t = actions_t.unsqueeze(0)        # shape: (1 env, episode_length, action_dim)
            next_states_t = torch.cat(next_states, dim=0)  # shape: (episode_length, state_space)
            next_states_t = next_states_t.unsqueeze(0)    # shape: (1 env, episode_length, state_space)
            rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)  # shape: (episode_length, 1)
            rewards_t = rewards_t.unsqueeze(0)        # shape: (1 env, episode_length, 1)

            dones_t = torch.zeros_like(rewards_t)  # shape: (1 env, episode_length, 1)
            dones_t[:, -1, :] = 1.0  # Mark the last step as done

            terminateds_t = torch.zeros_like(rewards_t)  # shape: (1 env, episode_length, 1)
            terminateds_t[:, -1, :] = terminated == True # Mark the last step as terminated or not
            truncateds_t = torch.zeros_like(rewards_t)  # shape: (1 env, episode_length, 1)
            truncateds_t[:, -1, :] = truncated == True # Mark the last step as truncated or not

            rollout_data = ExperienceData(
                states=states_t,
                actions=actions_t,
                next_states=next_states_t,
                rewards=rewards_t,
                terminateds=terminateds_t,
                truncateds=truncateds_t,
                dones=dones_t,
            )

            return rollout_data
    
    def collect_rollout_data(
        self, envs: gym.vector.VectorEnv, initial_states: torch.Tensor, agent: AgentProtocol, tracker: SessionTracker
        ) -> tuple[RolloutData, list[np.ndarray]]:
            states = []
            actions = []
            next_states = []
            rewards = []
            critic_values = []
            log_probs = []

            env = envs.envs[0]

            state, _ = env.reset()

            while not tracker.is_session_complete():
                tracker.increment_timestep()

                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # shape: (1, state_space)

                with torch.no_grad():
                    action_t, _ = agent.actor.output(state_t)
                    # apply exploration noise etc to the actions
                    action_t = agent.actor.exploration(action_t)
                    log_prob, _ = agent.actor.get_log_prob_and_entropy(state_t, action_t)
                    critic_value = agent.get_critic_value(state_t, action_t)

                action_np = action_t.squeeze().numpy()
                next_state, reward, terminated, truncated, info = env.step(action_np)

                next_state_t = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)  # shape: (1, state_space)
                
                states.append(state_t)
                actions.append(action_t)
                next_states.append(next_state_t)
                rewards.append(reward)
                critic_values.append(critic_value)
                log_probs.append(log_prob)
                
                state = next_state
                
                if terminated or truncated:
                    episode_return = info['episode']['r']
                    episode_length = info['episode']['l']

                    tracker.finish_episode(episode_return, episode_length)
                    break

            states_t = torch.cat(states, dim=0)        # shape: (episode_length, state_space)
            # add an additional dimension for n_envs at dim 0
            states_t = states_t.unsqueeze(0)          # shape: (1 env, episode_length, state_space)
            actions_t = torch.cat(actions, dim=0)      # shape: (episode_length, action_dim)
            actions_t = actions_t.unsqueeze(0)        # shape: (1 env, episode_length, action_dim)
            next_states_t = torch.cat(next_states, dim=0)  # shape: (episode_length, state_space)
            next_states_t = next_states_t.unsqueeze(0)    # shape: (1 env, episode_length, state_space)
            rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)  # shape: (episode_length, 1)
            rewards_t = rewards_t.unsqueeze(0)        # shape: (1 env, episode_length, 1)

            dones_t = torch.zeros_like(rewards_t)  # shape: (1 env, episode_length, 1)
            dones_t[:, -1, :] = 1.0  # Mark the last step as done

            terminateds_t = torch.zeros_like(rewards_t)  # shape: (1 env, episode_length, 1)
            terminateds_t[:, -1, :] = terminated == True # Mark the last step as terminated or not
            truncateds_t = torch.zeros_like(rewards_t)  # shape: (1 env, episode_length, 1)
            truncateds_t[:, -1, :] = truncated == True # Mark the last step as truncated or not

            critic_values_t = torch.cat(critic_values, dim=0)  # shape: (episode_length, 1)
            critic_values_t = critic_values_t.unsqueeze(0)    # shape: (1 env, episode_length, 1)
            log_probs_t = torch.cat(log_probs, dim=0)  # shape: (episode_length, 1)
            log_probs_t = log_probs_t.unsqueeze(0)    # shape: (1 env, episode_length, 1)

            returns_t = torch.zeros_like(rewards_t)  # shape: (1 env, episode_length, 1)
            advantages_t = torch.zeros_like(rewards_t)  # shape: (1 env, episode_length, 1)

            rollout_data = RolloutData(
                states=states_t,
                actions=actions_t,
                next_states=next_states_t,
                rewards=rewards_t,
                terminateds=terminateds_t,
                truncateds=truncateds_t,
                dones=dones_t,
                critic_values=critic_values_t,
                log_probs=log_probs_t,
                returns=returns_t,
                advantages=advantages_t
            )

            return rollout_data, [next_state]