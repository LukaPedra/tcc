import time
import torch
import numpy as np  # Necessário para converter LazyFrames
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack
from pathlib import Path
import sys

# Adiciona o diretório atual ao path para imports funcionarem
file_path = Path(__file__).resolve()
sys.path.append(str(file_path.parent))

from wrappers import SkipFrame, GrayScaleObservation, ResizeObservation
from agent import PPOAgent

def play_game(checkpoint_path):
    # 1. Configurar Ambiente
    if gym.__version__ < '0.26':
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
    else:
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='human', apply_api_compatibility=True)

    env = JoypadSpace(env, [["right"], ["right", "A"]])
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    if gym.__version__ < '0.26':
        env = FrameStack(env, num_stack=4, new_step_api=True)
    else:
        env = FrameStack(env, num_stack=4)

    # 2. Inicializar Agente
    # save_dir não importa para replay
    mario = PPOAgent(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=Path("."))

    # 3. Carregar Modelo
    print(f"Carregando PPO: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=mario.device)
        if 'model' in checkpoint:
            mario.policy.load_state_dict(checkpoint['model'])
            steps = checkpoint.get('steps', 'Desconhecido')
            print(f"Modelo carregado (Steps: {steps})")
        else:
            mario.policy.load_state_dict(checkpoint)
            print("Modelo carregado (Formato antigo)")
    except Exception as e:
        print(f"Erro ao carregar o ficheiro: {e}")
        return

    mario.policy.eval()

    state = env.reset()
    total_reward = 0

    print("\nIniciando replay... (Pressione Ctrl+C para parar)")

    try:
        while True:
            env.render()

            # --- CORREÇÃO DO ESTADO ---
            # Verifica se state é uma tupla (reset) ou objeto direto (step)
            if isinstance(state, tuple):
                state_obs = state[0]
            else:
                state_obs = state

            # Converte LazyFrames para array numpy e depois para tensor
            state_np = np.array(state_obs)
            state_tensor = torch.tensor(state_np, device=mario.device).unsqueeze(0)

            # Obter ação determinística (argmax)
            with torch.no_grad():
                dist, _ = mario.policy(state_tensor)
                action = torch.argmax(dist.probs).item()

            next_state, reward, done, trunc, info = env.step(action)
            total_reward += reward
            state = next_state # Atualiza state para o próximo loop

            time.sleep(0.05) # Controlo de FPS

            if done or info.get("flag_get", False):
                print(f"Fim do episódio! Reward Total: {total_reward}")
                break

    except KeyboardInterrupt:
        print("Replay interrompido.")
    finally:
        env.close()

if __name__ == "__main__":
    # Procura automaticamente o checkpoint mais recente na pasta checkpoints_ppo
    checkpoint_dir = Path("checkpoints_ppo")

    if not checkpoint_dir.exists():
        # Tenta procurar na raiz se estiver a rodar de dentro da pasta PPO
        checkpoint_dir = Path("../checkpoints_ppo")

    try:
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Pasta {checkpoint_dir} não encontrada.")

        # Encontra o arquivo .chkpt mais recente
        files = list(checkpoint_dir.glob("**/*.chkpt"))
        if not files:
            raise FileNotFoundError("Nenhum arquivo .chkpt encontrado.")

        latest = max(files, key=lambda p: p.stat().st_mtime)
        play_game(latest)

    except Exception as e:
        print(f"Erro: {e}")
        print("Verifique se treinou o modelo e se a pasta 'checkpoints_ppo' existe.")