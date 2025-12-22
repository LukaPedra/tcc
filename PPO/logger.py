import numpy as np
import time, datetime
import matplotlib.pyplot as plt

class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log_ppo.txt"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanEntropy':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )

        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_entropies_plot = save_dir / "entropy_plot.jpg"

        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_losses = []
        self.ep_entropies = []

        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_losses = []
        self.moving_avg_ep_entropies = []

        self.init_episode()
        self.record_time = time.time()

    def log_step(self, reward, loss, entropy):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss is not None:
            self.curr_ep_loss += loss
            self.curr_ep_entropy += entropy
            self.curr_ep_loss_count += 1

    def log_episode(self):
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)

        if self.curr_ep_loss_count > 0:
            avg_loss = self.curr_ep_loss / self.curr_ep_loss_count
            avg_entropy = self.curr_ep_entropy / self.curr_ep_loss_count
        else:
            avg_loss = 0 if len(self.ep_losses) == 0 else self.ep_losses[-1]
            avg_entropy = 0 if len(self.ep_entropies) == 0 else self.ep_entropies[-1]

        self.ep_losses.append(avg_loss)
        self.ep_entropies.append(avg_entropy)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_entropy = 0.0
        self.curr_ep_loss_count = 0

    # CORREÇÃO: Adicionado argumento 'step'
    def record(self, episode, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_losses[-100:]), 3)
        mean_ep_entropy = np.round(np.mean(self.ep_entropies[-100:]), 3)

        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_losses.append(mean_ep_loss)
        self.moving_avg_ep_entropies.append(mean_ep_entropy)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - " # <--- Agora imprime o passo correto
            f"Reward {mean_ep_reward} - "
            f"Length {mean_ep_length} - "
            f"Loss {mean_ep_loss} - "
            f"Entropy {mean_ep_entropy} - "
            f"TimeDelta {time_since_last_record}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{mean_ep_reward:15.3f}{mean_ep_length:15.3f}" # <--- E grava no ficheiro
                f"{mean_ep_loss:15.3f}{mean_ep_entropy:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_losses", "ep_entropies"]:
            plt.clf()
            plt.plot(getattr(self, f"moving_avg_{metric}"), label=metric)
            plt.legend()
            plt.savefig(getattr(self, f"{metric}_plot"))