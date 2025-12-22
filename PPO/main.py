import os
import datetime
from pathlib import Path
import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
from wrappers import SkipFrame, GrayScaleObservation, ResizeObservation, CustomReward
from agent import PPOAgent
from logger import MetricLogger

# 1. Configurar Ambiente
if gym.__version__ < '0.26':
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
else:
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='rgb', apply_api_compatibility=True)

env = JoypadSpace(env, [["right"], ["right", "A"]])
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = CustomReward(env)  # <--- Adicione esta linha antes do FrameStack
if gym.__version__ < '0.26':
    env = FrameStack(env, num_stack=4, new_step_api=True)
else:
    env = FrameStack(env, num_stack=4)

env.reset()

# 2. Inicializar Logging
save_dir = Path("checkpoints_ppo") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)
logger = MetricLogger(save_dir)

# 3. Inicializar Agente
mario = PPOAgent(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

# 4. Training Loop
print("Iniciando Treino PPO...")

episodes = 40000
total_steps = 0  # <--- NOVA VARIÁVEL para contar passos globais

for e in range(episodes):
    state = env.reset()

    while True:
        total_steps += 1 # <--- Incrementa a cada passo

        # PPO: Actor retorna ação, log_prob e valor estimado
        action, log_prob, val = mario.act(state)

        # Passo no ambiente
        next_state, reward, done, trunc, info = env.step(action)

        # Ajuste do state para formato correto antes de guardar
        state_np = state[0].__array__() if isinstance(state, tuple) else state.__array__()

        # Guardar na memória
        mario.remember(state_np, action, log_prob, val, reward, done)

        # Tentar aprender (só acontece se buffer encher)
        loss, entropy = mario.learn()

        # Logging (se houve update, loss/entropy não serão None)
        logger.log_step(reward, loss, entropy)

        state = next_state

        if done or info["flag_get"]:
            break

    logger.log_episode()

    if e % 20 == 0:
        # CORREÇÃO: Agora passamos 'step=total_steps'
        logger.record(episode=e, step=total_steps)