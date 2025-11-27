import argparse
import os
import time
import numpy as np
import cv2
import torch
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from .utils import create_mario_env
from .ppo import PPOAgent


def train_ppo(
    world=1,
    stage=1,
    total_timesteps=5000,
    rollout_steps=1024,
    render=False,
    save_path=None,
    save_interval=1000,
):
    env = create_mario_env(world=world, stage=stage)
    # restrict action space to SIMPLE_MOVEMENT mapping
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    obs_shape = env.observation_space.shape  # (C,H,W)
    # Some environments / wrappers (and Gym vs Gymnasium differences) may not expose
    # a reliable `.n` attribute on the action space. Fall back to the SIMPLE_MOVEMENT
    # length which JoypadSpace is built from.
    n_actions = getattr(env.action_space, "n", None)
    if n_actions is None:
        try:
            from gym_super_mario_bros.actions import SIMPLE_MOVEMENT as _SM

            n_actions = len(_SM)
        except Exception:
            # Last-resort fallback: try to infer from action_space shape
            space = env.action_space
            if hasattr(space, "n") and space.n is not None:
                n_actions = space.n
            else:
                # default to 1 if nothing else works
                n_actions = 1

    agent = PPOAgent(obs_shape, n_actions)

    timesteps = 0
    episode_rewards = []
    episode_reward = 0.0
    obs = env.reset()
    last_saved = 0

    # prepare save directory
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    def evaluate_and_record(agent, step, num_episodes=1, max_steps_per_episode=5000, fps=30):
        """Run deterministic evaluation episodes and save videos to save_path/videos.
        Uses agent.net directly to pick argmax actions (deterministic).
        """
        vid_dir = os.path.join(save_path, "videos") if save_path else None
        if vid_dir:
            os.makedirs(vid_dir, exist_ok=True)

        eval_env = create_mario_env(world=world, stage=stage)
        eval_env = JoypadSpace(eval_env, SIMPLE_MOVEMENT)

        rewards = []
        for ep in range(num_episodes):
            frames = []
            obs = eval_env.reset()
            ep_reward = 0.0
            done = False
            steps = 0
            while not done and steps < max_steps_per_episode:
                # render frame
                try:
                    frame = eval_env.render(mode="rgb_array")
                except TypeError:
                    # older gym might not accept mode kwarg
                    frame = eval_env.render()
                if frame is not None:
                    frames.append(frame)

                # deterministic action: argmax logits
                with torch.no_grad():
                    obs_t = agent._to_tensor(obs)
                    logits, _ = agent.net(obs_t)
                    action = int(torch.argmax(logits, dim=-1).item())

                next_obs, reward, done, info = eval_env.step(action)
                ep_reward += float(reward)
                obs = next_obs
                steps += 1

            rewards.append(ep_reward)

            # save video for this episode
            if vid_dir and len(frames) > 0:
                # build video writer
                h, w = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_path = os.path.join(vid_dir, f"ckpt_{step}_ep{ep}.mp4")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                for f in frames:
                    # frames are RGB, convert to BGR for OpenCV
                    writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                writer.release()
                print(f"Saved eval video: {out_path} (reward={ep_reward:.2f})")

        eval_env.close()
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        print(f"Eval mean reward @ {step}: {mean_reward:.2f}")
        return mean_reward

    while timesteps < total_timesteps:
        # collect rollout
        steps = 0
        while steps < rollout_steps and timesteps < total_timesteps:
            state = np.array(obs)
            action, logp, value = agent.act(state)
            next_obs, reward, done, info = env.step(action)

            agent.store(state, action, logp, value, reward, done)

            episode_reward += reward
            timesteps += 1
            steps += 1

            if render:
                env.render()

            if done:
                episode_rewards.append(episode_reward)
                print(f"Episode done - reward={episode_reward:.2f} timesteps={timesteps}")
                episode_reward = 0.0
                next_obs = env.reset()

            obs = next_obs

        # update agent
        # provide last obs for value bootstrap
        last_state = np.array(obs)
        stats = agent.update(last_obs=last_state)

        # checkpointing
        if save_path and (timesteps - last_saved) >= save_interval:
            ckpt_path = os.path.join(save_path, f"ppo_{timesteps}.pt")
            agent.save(ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
            last_saved = timesteps
            # run evaluation and record video for this checkpoint
            try:
                evaluate_and_record(agent, timesteps, num_episodes=1)
            except Exception as e:
                print("Eval/recording failed:", e)

        # print training metrics from update (if returned)
        if isinstance(stats, dict):
            print(f"Update stats: actor_loss={stats.get('actor_loss', 0):.4f}, value_loss={stats.get('value_loss',0):.4f}, entropy={stats.get('entropy',0):.4f}")

        # quick status
        if len(episode_rewards) > 0:
            avg_r = sum(episode_rewards[-10:]) / min(10, len(episode_rewards))
        else:
            avg_r = 0.0
        print(f"Timesteps: {timesteps}/{total_timesteps}  avg_recent_reward={avg_r:.2f}")

    if save_path:
        final_path = os.path.join(save_path, "ppo_final.pt")
        agent.save(final_path)
        print(f"Saved final model: {final_path}")

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=5000)
    parser.add_argument('--rollout', type=int, default=1024)
    parser.add_argument('--world', type=int, default=1)
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--save-interval', type=int, default=1000)

    args = parser.parse_args()
    train_ppo(
        world=args.world,
        stage=args.stage,
        total_timesteps=args.timesteps,
        rollout_steps=args.rollout,
        render=args.render,
        save_path=args.save,
        save_interval=args.save_interval,
    )