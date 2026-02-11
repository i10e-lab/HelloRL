import numpy as np
import torch
import gymnasium as gym
from PIL import Image, ImageDraw, ImageFont
import IPython.display as ipythondisplay
import time

gym.envs.registration.register(
    id='LunarLanderUpgraded-v1',
    entry_point='helloRL.utils.lunar_lander_upgraded:LunarLander',
    max_episode_steps=1000,
)

def env_theme_for_env(env_name):
    if 'Lander' in env_name:
        return 'dark'
    else:
        return 'light'
    
def episode_complete(terminated, truncated):
    return terminated or truncated
    
# different environments have different success criteria
# for example cartpole can complete unsuccessfully with terminated True and reward > 0
# while lunar lander can complete successfully with terminated True and reward > 0
# so this function verifies if the episode is successful
def success_for_completed_episode(env_name, reward, terminated, truncated):
    if 'CartPole' in env_name:
        return truncated
    elif 'Lander' in env_name:
        return reward == 100

# render a frame for the environment inline in a jupyter notebook
def render_env(env, reward, terminated, truncated, title=None):
    img = env.render()

    overlay_color = None

    if episode_complete(terminated, truncated):
        success = success_for_completed_episode(env.spec.id, reward, terminated, truncated)
        
        if success:
            overlay_color = (0, 255, 0, 128)
        else:
            overlay_color = (255, 0, 0, 128)

    if overlay_color:
        overlay = Image.new('RGBA', img.shape[1::-1], overlay_color)
        frame_with_overlay = Image.alpha_composite(Image.fromarray(img).convert('RGBA'), overlay)
        pil_img = frame_with_overlay.convert('RGB')
    else:
        pil_img = Image.fromarray(img)

    if title is not None:
        draw = ImageDraw.Draw(pil_img)

        font = ImageFont.truetype("Inter.ttc", 24, index=10)
        position = (10, 10)
        env_theme = env_theme_for_env(env.spec.id)
        
        if env_theme == 'light':
            draw.text(position, title, fill=(0,0,0), font=font)
        else:
            draw.text(position, title, fill=(255,255,255), font=font)

    ipythondisplay.display(pil_img)
    ipythondisplay.clear_output(wait=True)

def run_sim_once(env, agent, seed=None, options=None, render_params=None):
    state, _ = env.reset(seed=seed, options=options)

    episode_rewards = []   # List to store rewards for current episode

    while True:
        # We now get action, log_prob, and critic_value from the agent
        state_t = torch.tensor(state)

        frame_title = None

        # if agent has `get_action` function, use it, otherwise use `output`
        if hasattr(agent, 'get_action_with_title'):
            action_t, frame_title = agent.get_action_with_title(state_t, len(episode_rewards))
        elif hasattr(agent, 'get_action'):
            action_t = agent.get_action(state_t)
        else:
            action_t = agent.output(state_t)[0]

        action_np = action_t.squeeze().detach().numpy()
        next_state, reward, terminated, truncated, info = env.step(action_np)

        episode_rewards.append(reward)
        
        state = next_state

        complete = episode_complete(terminated, truncated)

        if render_params is not None:
            title = render_params.get('title', None)

            if frame_title is not None:
                title = f'{title}, {frame_title}' if title else frame_title

            start = time.time()
                
            render_env(env, reward, terminated, truncated, title=title)

            if complete:
                time.sleep(1)
            else:
                fps = env.metadata.get('render_fps', 50)
                end = time.time()
                elapsed = end - start
                sleep_time = max(0, (1 / fps) - elapsed)
                time.sleep(sleep_time)
        
        if complete:
            break
    
    episode_score = sum(episode_rewards)

    return episode_score


# run sim with env
# env will need to have render mode set to 'rgb_array' in order to render inline in a jupyter notebook
def run_sim_with_env(env, agent, episodes=1, render_params=None, seed=None):
    scores = [] # To keep track of episode scores for plotting

    _, _ = env.reset(seed=seed)

    for _ in range(episodes):
        episode_score = run_sim_once(env, agent, render_params=render_params)
        scores.append(episode_score)

    env.close()
    
    return scores, np.mean(scores).item()  # Return scores and average score

# run the sim, with optional rendering inline in a jupyter notebook
# render_params can be None or a dict with optional keys:
# - title: a string to display as the title of the rendered frame
# returns a tuple of scores for each episode and the average score
def run_sim(env_name, agent, episodes=1, render_params=None, seed=None):
    if render_params is not None:
        env = gym.make(env_name, render_mode='rgb_array')
    else:
        env = gym.make(env_name)

    return run_sim_with_env(env, agent, episodes=episodes, render_params=render_params, seed=seed)