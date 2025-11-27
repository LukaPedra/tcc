#!/usr/bin/env python3
"""Run a trained PPO checkpoint for SuperMarioBros.

Usage examples:
  .venv/bin/python3 run_trained.py --checkpoint models/ppo_quick/ppo_final.pt --episodes 3
  .venv/bin/python3 run_trained.py --checkpoint models/ppo_quick/ppo_final.pt --episodes 1 --no-render --save-video --video-path models/ppo_quick/run.mp4
"""
import argparse
import os
import cv2
import time
import torch
import numpy as np
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from mario.utils import create_mario_env
from mario.ppo import PPOAgent


def run_episode(env, agent, render=True, deterministic=True, max_steps=10000, save_video=False, realtime=True, fps=30):
    obs = env.reset()
    total_reward = 0.0
    done = False
    steps = 0
    frames = []

    while not done and steps < max_steps:
        # capture frame if requested
        frame_start = time.time()
        if save_video:
            try:
                frame = env.render(mode="rgb_array")
            except TypeError:
                frame = env.render()
            if frame is not None:
                frames.append(frame)

        # action selection (deterministic argmax or sample)
        obs_t = agent._to_tensor(obs)
        with torch.no_grad():
            logits, _ = agent.net(obs_t)
        if deterministic:
            action = int(torch.argmax(logits, dim=-1).item())
        else:
            probs = torch.softmax(logits, dim=-1)
            action = int(torch.multinomial(probs, num_samples=1).item())

        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if render and not save_video:
            env.render()

        # throttle loop to real-time fps if requested
        if realtime:
            elapsed = time.time() - frame_start
            target = 1.0 / float(fps)
            to_sleep = target - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

    return total_reward, frames


def save_frames_as_video(frames, path, fps=30):
    if len(frames) == 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def main(args):
    env = create_mario_env(world=args.world, stage=args.stage)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    obs_shape = env.observation_space.shape
    n_actions = getattr(env.action_space, "n", None)
    if n_actions is None:
        n_actions = len(SIMPLE_MOVEMENT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    agent = PPOAgent(obs_shape, n_actions, device=device)
    agent.load(args.checkpoint, map_location=device)
    agent.net.eval()

    for ep in range(args.episodes):
        r, frames = run_episode(
            env,
            agent,
            render=not args.no_render,
            deterministic=not args.stochastic,
            max_steps=args.max_steps,
            save_video=args.save_video,
            realtime=not args.no_realtime,
            fps=args.fps,
        )
        print(f"Episode {ep+1}/{args.episodes} reward: {r:.2f}")
        if args.save_video and frames:
            out_path = args.video_path or os.path.join(os.path.dirname(args.checkpoint), f"run_ep{ep+1}.mp4")
            save_frames_as_video(frames, out_path, fps=args.fps)
            print("Saved video:", out_path)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model .pt file (state_dict)")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--no-render", action="store_true", help="Do not render to screen")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions stochastically instead of argmax")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--save-video", action="store_true", help="Save episode frames to mp4 instead of rendering live")
    parser.add_argument("--video-path", type=str, default=None, help="Path to write video file")
    parser.add_argument("--fps", type=int, default=30, help="FPS for saved video")
    parser.add_argument("--no-realtime", action="store_true", help="Do not throttle playback to realtime; run as fast as possible")
    args = parser.parse_args()

    main(args)
