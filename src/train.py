import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import DQNAgent
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo  # RecordVideoクラスをインポート

# デバイスの設定
device = torch.device("cpu")

# 環境の初期化
env = gym.make('CartPole-v1', render_mode='rgb_array')  # render_modeを設定
env = RecordVideo(env, './video', episode_trigger=lambda episode_id: episode_id % 100 == 0)  # 100エピソードごとに動画を記録
env._max_episode_steps = 1000  # 最大ステップ数を1000に設定
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
num_episodes = 10000
agent = DQNAgent(state_size=state_size, action_size=action_size, device=device)

# ターゲットネットワークの更新頻度を設定
agent.update_target_every = 1000  # 例として1000ステップごとに更新

# バッチサイズの設定
batch_size = 32

# 報酬を記録するリストの追加
rewards = []

# トレーニングループ
for episode in range(num_episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float, device=device)
    done = False
    total_reward = 0

    while not done:
        # 行動の選択
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = torch.tensor(next_state, dtype=torch.float, device=device)
        
        # エージェントの経験を記録
        agent.remember(state, action, reward, next_state, done)
        
        # 状態の更新
        state = next_state
        total_reward += reward
        
        # エージェントの学習
        agent.replay(batch_size)

    rewards.append(total_reward)  # エピソード終了時に報酬を記録

    print(f"Episode: {episode + 1}/{num_episodes}, Score: {total_reward:*>5}, Epsilon: {agent.epsilon:.4f}")

# 学習の進捗をグラフに表示
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress')
plt.savefig('training_progress.png')  # グラフを保存
plt.close()

# モデルの保存
torch.save(agent.model.state_dict(), 'dqn_cartpole.pth')
env.close()