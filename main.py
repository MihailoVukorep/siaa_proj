import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from typing import Tuple, Dict, List

class CartPoleEnvironment:
    """
    Implementacija Cart-Pole okruženja na osnovu matematičkog modela
    """
    
    def __init__(self, 
                 m: float = 0.1,      # masa štapa (100g)
                 M: float = 1.1,      # ukupna masa štapa i kolica (1.1kg)
                 l: float = 0.5,      # dužina sipke (0.5m)
                 g: float = 9.81,     # gravitaciona konstanta
                 k: float = 1/3,      # konstanta za moment inercije
                 T: float = 0.02,     # vreme odabiranja (20ms)
                 theta_th: float = 0.2094,  # granična vrednost ugla (12°)
                 x_th: float = 2.4,   # granična pozicija
                 max_steps: int = 500):
        
        self.m = m
        self.M = M
        self.l = l
        self.g = g
        self.k = k
        self.T = T
        self.theta_th = theta_th
        self.x_th = x_th
        self.max_steps = max_steps
        
        # Diskretni prostor akcija: [-10N, 0N, +10N]
        self.actions = [-10.0, 0.0, 10.0]
        self.n_actions = len(self.actions)
        
        # Diskretizacija prostora stanja
        self.x_bins = np.linspace(-2.4, 2.4, 15)
        self.x_dot_bins = np.linspace(-3.0, 3.0, 15)
        self.theta_bins = np.linspace(-0.3, 0.3, 15)
        self.theta_dot_bins = np.linspace(-2.0, 2.0, 15)
        
        self.reset()
    
    def reset(self) -> Tuple[int, int, int, int]:
        """Reset okruženja na početno stanje"""
        self.state = np.array([
            np.random.uniform(-0.1, 0.1),  # x
            np.random.uniform(-0.1, 0.1),  # x_dot
            np.random.uniform(-0.1, 0.1),  # theta
            np.random.uniform(-0.1, 0.1)   # theta_dot
        ])
        self.steps = 0
        return self._discretize_state()
    
    def _discretize_state(self) -> Tuple[int, int, int, int]:
        """Diskretizacija kontinualnog stanja"""
        x, x_dot, theta, theta_dot = self.state
        
        x_idx = np.digitize(x, self.x_bins) - 1
        x_dot_idx = np.digitize(x_dot, self.x_dot_bins) - 1
        theta_idx = np.digitize(theta, self.theta_bins) - 1
        theta_dot_idx = np.digitize(theta_dot, self.theta_dot_bins) - 1
        
        # Ograničavanje na validne indekse
        x_idx = np.clip(x_idx, 0, len(self.x_bins) - 2)
        x_dot_idx = np.clip(x_dot_idx, 0, len(self.x_dot_bins) - 2)
        theta_idx = np.clip(theta_idx, 0, len(self.theta_bins) - 2)
        theta_dot_idx = np.clip(theta_dot_idx, 0, len(self.theta_dot_bins) - 2)
        
        return (x_idx, x_dot_idx, theta_idx, theta_dot_idx)
    
    def _f_theta(self, z: np.ndarray, F: float) -> float:
        """Funkcija za ugaonu acceleraciju"""
        x, x_dot, theta, theta_dot = z
        
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        numerator = (self.M * self.g * sin_theta - 
                    cos_theta * (F + self.m * self.l * theta_dot**2 * sin_theta))
        denominator = ((1 + self.k) * self.M * self.l - 
                      self.m * self.l * cos_theta**2)
        
        return numerator / denominator
    
    def _f_x(self, z: np.ndarray, F: float) -> float:
        """Funkcija za linearnu acceleraciju"""
        x, x_dot, theta, theta_dot = z
        
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        numerator = (self.m * self.g * sin_theta * cos_theta - 
                    (1 + self.k) * (F + self.m * self.l * theta_dot**2 * sin_theta))
        denominator = (self.m * cos_theta**2 - (1 + self.k) * self.M)
        
        return numerator / denominator
    
    def step(self, action_idx: int) -> Tuple[Tuple[int, int, int, int], float, bool]:
        """Izvršavanje koraka u okruženju"""
        F = self.actions[action_idx]
        
        # Euler diskretizacija
        x, x_dot, theta, theta_dot = self.state
        
        # Računanje novih vrednosti
        x_new = x + self.T * x_dot
        x_dot_new = x_dot + self.T * self._f_x(self.state, F)
        theta_new = theta + self.T * theta_dot
        theta_dot_new = theta_dot + self.T * self._f_theta(self.state, F)
        
        self.state = np.array([x_new, x_dot_new, theta_new, theta_dot_new])
        self.steps += 1
        
        # Provera terminalnog stanja
        done = (abs(theta_new) >= self.theta_th or 
                abs(x_new) >= self.x_th or 
                self.steps >= self.max_steps)
        
        # Računanje nagrade (opcija 3 iz zadatka)
        if done and self.steps < self.max_steps:
            reward = -100  # Velika kazna za pad
        else:
            reward = 1  # Pozitivna nagrada za svaki korak
        
        return self._discretize_state(), reward, done


class QLearningAgent:
    """
    Q-Learning agent za cart-pole problem
    """
    
    def __init__(self, 
                 n_actions: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-tabela kao default dictionary
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        
        # Statistike treniranja
        self.episode_rewards = []
        self.episode_lengths = []
    
    def choose_action(self, state: Tuple[int, int, int, int]) -> int:
        """Epsilon-greedy izbor akcije"""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state: Tuple[int, int, int, int], 
               action: int, 
               reward: float, 
               next_state: Tuple[int, int, int, int], 
               done: bool):
        """Q-learning update"""
        current_q = self.q_table[state][action]
        
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        # Q-learning update formula
        self.q_table[state][action] += self.learning_rate * (target - current_q)
    
    def decay_epsilon(self):
        """Smanjivanje epsilon vrednosti"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class SARSAAgent:
    """
    SARSA agent za cart-pole problem
    """
    
    def __init__(self, 
                 n_actions: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-tabela kao default dictionary
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        
        # Statistike treniranja
        self.episode_rewards = []
        self.episode_lengths = []
    
    def choose_action(self, state: Tuple[int, int, int, int]) -> int:
        """Epsilon-greedy izbor akcije"""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state: Tuple[int, int, int, int], 
               action: int, 
               reward: float, 
               next_state: Tuple[int, int, int, int], 
               next_action: int, 
               done: bool):
        """SARSA update"""
        current_q = self.q_table[state][action]
        
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * self.q_table[next_state][next_action]
        
        # SARSA update formula
        self.q_table[state][action] += self.learning_rate * (target - current_q)
    
    def decay_epsilon(self):
        """Smanjivanje epsilon vrednosti"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_q_learning(env: CartPoleEnvironment, 
                    agent: QLearningAgent, 
                    episodes: int = 5000) -> QLearningAgent:
    """Treniranje Q-Learning agenta"""
    
    print("Treniranje Q-Learning agenta...")
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        agent.episode_rewards.append(episode_reward)
        agent.episode_lengths.append(episode_length)
        agent.decay_epsilon()
        
        if episode % 500 == 0:
            avg_reward = np.mean(agent.episode_rewards[-100:])
            avg_length = np.mean(agent.episode_lengths[-100:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return agent


def train_sarsa(env: CartPoleEnvironment, 
                agent: SARSAAgent, 
                episodes: int = 5000) -> SARSAAgent:
    """Treniranje SARSA agenta"""
    
    print("Treniranje SARSA agenta...")
    
    for episode in range(episodes):
        state = env.reset()
        action = agent.choose_action(state)
        episode_reward = 0
        episode_length = 0
        
        while True:
            next_state, reward, done = env.step(action)
            
            if not done:
                next_action = agent.choose_action(next_state)
            else:
                next_action = 0  # Vrednost nije bitna jer je done=True
            
            agent.update(state, action, reward, next_state, next_action, done)
            
            state = next_state
            action = next_action
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        agent.episode_rewards.append(episode_reward)
        agent.episode_lengths.append(episode_length)
        agent.decay_epsilon()
        
        if episode % 500 == 0:
            avg_reward = np.mean(agent.episode_rewards[-100:])
            avg_length = np.mean(agent.episode_lengths[-100:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return agent


def test_agent(env: CartPoleEnvironment, agent, episodes: int = 10):
    """Testiranje naučenog agenta"""
    
    print(f"\nTestiranje agenta na {episodes} epizoda...")
    
    test_rewards = []
    test_lengths = []
    
    # Privremeno isključiti exploration
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        test_rewards.append(episode_reward)
        test_lengths.append(episode_length)
        
        print(f"Test Episode {episode + 1}: Reward = {episode_reward}, Length = {episode_length}")
    
    # Vrati originalnu epsilon vrednost
    agent.epsilon = original_epsilon
    
    avg_reward = np.mean(test_rewards)
    avg_length = np.mean(test_lengths)
    
    print(f"\nTest Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Episode Length: {avg_length:.2f}")
    print(f"Success Rate: {sum(1 for r in test_rewards if r > 400) / len(test_rewards) * 100:.1f}%")
    
    return test_rewards, test_lengths


def plot_results(q_agent: QLearningAgent, sarsa_agent: SARSAAgent):
    """Crtanje grafikona rezultata"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Q-Learning rewards
    window_size = 100
    q_rewards_smooth = np.convolve(q_agent.episode_rewards, 
                                  np.ones(window_size)/window_size, 
                                  mode='valid')
    ax1.plot(q_rewards_smooth, label='Q-Learning', color='blue', alpha=0.7)
    ax1.set_title('Q-Learning - Episode Rewards (Moving Average)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    ax1.legend()
    
    # SARSA rewards
    sarsa_rewards_smooth = np.convolve(sarsa_agent.episode_rewards, 
                                      np.ones(window_size)/window_size, 
                                      mode='valid')
    ax2.plot(sarsa_rewards_smooth, label='SARSA', color='red', alpha=0.7)
    ax2.set_title('SARSA - Episode Rewards (Moving Average)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward')
    ax2.grid(True)
    ax2.legend()
    
    # Poređenje rewards
    ax3.plot(q_rewards_smooth, label='Q-Learning', color='blue', alpha=0.7)
    ax3.plot(sarsa_rewards_smooth, label='SARSA', color='red', alpha=0.7)
    ax3.set_title('Comparison - Episode Rewards')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Reward')
    ax3.grid(True)
    ax3.legend()
    
    # Episode lengths
    q_lengths_smooth = np.convolve(q_agent.episode_lengths, 
                                  np.ones(window_size)/window_size, 
                                  mode='valid')
    sarsa_lengths_smooth = np.convolve(sarsa_agent.episode_lengths, 
                                      np.ones(window_size)/window_size, 
                                      mode='valid')
    
    ax4.plot(q_lengths_smooth, label='Q-Learning', color='blue', alpha=0.7)
    ax4.plot(sarsa_lengths_smooth, label='SARSA', color='red', alpha=0.7)
    ax4.set_title('Comparison - Episode Lengths')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Steps')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()


def simulate_episode(env: CartPoleEnvironment, agent, visualize: bool = False):
    """Simulacija jedne epizode sa opcijom vizualizacije"""
    
    state = env.reset()
    episode_reward = 0
    episode_length = 0
    
    # Čuvanje trajektorije za vizualizaciju
    trajectory = []
    
    # Privremeno isključiti exploration
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    while True:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        
        # Čuvanje stanja za vizualizaciju
        trajectory.append({
            'x': env.state[0],
            'theta': env.state[2],
            'action': env.actions[action],
            'reward': reward
        })
        
        state = next_state
        episode_reward += reward
        episode_length += 1
        
        if done:
            break
    
    # Vrati originalnu epsilon vrednost
    agent.epsilon = original_epsilon
    
    if visualize:
        plot_trajectory(trajectory)
    
    return episode_reward, episode_length, trajectory


def plot_trajectory(trajectory: List[Dict]):
    """Crtanje trajektorije cart-pole sistema"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    steps = range(len(trajectory))
    x_vals = [t['x'] for t in trajectory]
    theta_vals = [t['theta'] for t in trajectory]
    actions = [t['action'] for t in trajectory]
    rewards = [t['reward'] for t in trajectory]
    
    # Pozicija kolica
    ax1.plot(steps, x_vals, 'b-', linewidth=2)
    ax1.set_title('Cart Position')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Position (m)')
    ax1.grid(True)
    ax1.axhline(y=2.4, color='r', linestyle='--', alpha=0.5, label='Boundary')
    ax1.axhline(y=-2.4, color='r', linestyle='--', alpha=0.5)
    ax1.legend()
    
    # Ugao štapa
    ax2.plot(steps, np.array(theta_vals) * 180/np.pi, 'g-', linewidth=2)
    ax2.set_title('Pole Angle')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Angle (degrees)')
    ax2.grid(True)
    ax2.axhline(y=12, color='r', linestyle='--', alpha=0.5, label='Boundary')
    ax2.axhline(y=-12, color='r', linestyle='--', alpha=0.5)
    ax2.legend()
    
    # Primenjene akcije
    ax3.step(steps, actions, 'r-', linewidth=2, where='post')
    ax3.set_title('Applied Actions')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Force (N)')
    ax3.grid(True)
    ax3.set_ylim(-11, 11)
    
    # Nagrade
    ax4.plot(steps, rewards, 'm-', linewidth=2)
    ax4.set_title('Rewards')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Reward')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()


def main():
    """Glavna funkcija za izvršavanje eksperimenta"""
    
    print("=== Cart-Pole Reinforcement Learning ===")
    print("Implementacija Q-Learning i SARSA algoritma")
    print("=" * 45)
    
    # Kreiranje okruženja
    env = CartPoleEnvironment()
    
    # Kreiranje agenata
    q_agent = QLearningAgent(n_actions=env.n_actions)
    sarsa_agent = SARSAAgent(n_actions=env.n_actions)
    
    # Treniranje agenata
    print("\n1. TRENIRANJE AGENATA")
    print("-" * 30)
    
    q_agent = train_q_learning(env, q_agent, episodes=5000)
    sarsa_agent = train_sarsa(env, sarsa_agent, episodes=5000)
    
    # Testiranje agenata
    print("\n2. TESTIRANJE AGENATA")
    print("-" * 30)
    
    print("Q-Learning Agent:")
    q_test_rewards, q_test_lengths = test_agent(env, q_agent, episodes=10)
    
    print("\nSARSA Agent:")
    sarsa_test_rewards, sarsa_test_lengths = test_agent(env, sarsa_agent, episodes=10)
    
    # Prikaz rezultata
    print("\n3. REZULTATI")
    print("-" * 30)
    
    plot_results(q_agent, sarsa_agent)
    
    # Simulacija najboljih agenata
    print("\n4. SIMULACIJA EPIZODA")
    print("-" * 30)
    
    print("Q-Learning Agent - Simulacija epizode:")
    q_reward, q_length, q_trajectory = simulate_episode(env, q_agent, visualize=True)
    print(f"Reward: {q_reward}, Length: {q_length}")
    
    print("\nSARSA Agent - Simulacija epizode:")
    sarsa_reward, sarsa_length, sarsa_trajectory = simulate_episode(env, sarsa_agent, visualize=True)
    print(f"Reward: {sarsa_reward}, Length: {sarsa_length}")
    
    # Finalni izveštaj
    print("\n5. FINALNI IZVEŠTAJ")
    print("-" * 30)
    
    print("Q-Learning Results:")
    print(f"  - Final training reward: {np.mean(q_agent.episode_rewards[-100:]):.2f}")
    print(f"  - Test average reward: {np.mean(q_test_rewards):.2f}")
    print(f"  - Test average length: {np.mean(q_test_lengths):.2f}")
    print(f"  - Success rate: {sum(1 for r in q_test_rewards if r > 400) / len(q_test_rewards) * 100:.1f}%")
    
    print("\nSARSA Results:")
    print(f"  - Final training reward: {np.mean(sarsa_agent.episode_rewards[-100:]):.2f}")
    print(f"  - Test average reward: {np.mean(sarsa_test_rewards):.2f}")
    print(f"  - Test average length: {np.mean(sarsa_test_lengths):.2f}")
    print(f"  - Success rate: {sum(1 for r in sarsa_test_rewards if r > 400) / len(sarsa_test_rewards) * 100:.1f}%")


if __name__ == "__main__":
    main()
