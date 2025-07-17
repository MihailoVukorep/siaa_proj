import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
import time
from collections import deque
import seaborn as sns

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CartPoleEnvironment:
    """
    Stabilizovana implementacija Cart-Pole okruženja sa poboljšanom dinamikom.
    """
    
    def __init__(self, m=0.1, M=1.0, l=0.5, g=9.81, k=1/3, T=0.01, 
                 force_mag=10.0, theta_threshold=0.2094, x_threshold=2.4):
        # Fizički parametri
        self.m = m  # masa štapa (kg)
        self.M = M  # masa kolica (kg)
        self.total_mass = M + m
        self.l = l  # dužina šipke (m)
        self.g = g  # gravitaciona konstanta (m/s^2)
        self.k = k  # konstanta momenta inercije
        self.T = T  # manje vreme odabiranja za stabilnost
        self.force_mag = force_mag
        
        # Granične vrednosti
        self.theta_threshold = theta_threshold  # ~12 stepeni
        self.x_threshold = x_threshold
        
        # Prostor stanja i akcija
        self.action_space = 2
        self.state_space = 4
        
        # Numerička stabilnost
        self.max_velocity = 5.0
        self.max_angular_velocity = 10.0
        
        self.state = None
        self.reset()
        
    def reset(self):
        """Resetuje okruženje u manje nasumično početno stanje."""
        # Manje nasumičnosti za lakše učenje
        self.state = np.random.uniform(low=-0.02, high=0.02, size=(4,))
        return self.state.copy()
    
    def _get_derivatives(self, state, force):
        """Stabilizovana verzija računanja izvoda."""
        x, x_dot, theta, theta_dot = state
        
        # Ograničavanje brzina za numeričku stabilnost
        x_dot = np.clip(x_dot, -self.max_velocity, self.max_velocity)
        theta_dot = np.clip(theta_dot, -self.max_angular_velocity, self.max_angular_velocity)
        
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Poboljšana formulacija za stabilnost
        temp = (force + self.m * self.l * theta_dot**2 * sin_theta) / self.total_mass
        
        # Ugaona akceleracija
        numerator_theta = self.g * sin_theta - cos_theta * temp
        denominator_theta = self.l * (4.0/3.0 - self.m * cos_theta**2 / self.total_mass)
        
        # Dodavanje epsilon za numeričku stabilnost
        denominator_theta = np.clip(denominator_theta, 1e-6, None)
        f_theta = numerator_theta / denominator_theta
        
        # Linearna akceleracija
        f_x = temp - self.m * self.l * f_theta * cos_theta / self.total_mass
        
        return f_x, f_theta
    
    def step(self, action):
        """Poboljšana implementacija koraka."""
        # Konvertovanje diskretne akcije u silu
        force = self.force_mag if action == 1 else -self.force_mag
        
        # Trenutno stanje
        x, x_dot, theta, theta_dot = self.state
        
        # Stabilizacija sistema - manje steps za integraciju
        for _ in range(2):  # Podeli korak na pola
            # Računanje izvoda
            f_x, f_theta = self._get_derivatives(self.state, force)
            
            # Euler diskretizacija sa pola koraka
            half_T = self.T / 2
            
            new_x = x + half_T * x_dot
            new_x_dot = x_dot + half_T * f_x
            new_theta = theta + half_T * theta_dot
            new_theta_dot = theta_dot + half_T * f_theta
            
            # Ograničavanje brzina
            new_x_dot = np.clip(new_x_dot, -self.max_velocity, self.max_velocity)
            new_theta_dot = np.clip(new_theta_dot, -self.max_angular_velocity, self.max_angular_velocity)
            
            # Ažuriranje stanja
            self.state = np.array([new_x, new_x_dot, new_theta, new_theta_dot])
            x, x_dot, theta, theta_dot = self.state
        
        # Provera terminalnog stanja
        done = (abs(x) > self.x_threshold or 
                abs(theta) > self.theta_threshold)
        
        # Poboljšana reward funkcija
        reward = self._calculate_reward(self.state, done)
        
        return self.state.copy(), reward, done
    
    def _calculate_reward(self, state, done):
        """Optimizovana reward funkcija fokusirana na stabilnost."""
        x, x_dot, theta, theta_dot = state
        
        if done:
            return -100  # Umanjena kazna za terminalno stanje
        
        # Reward komponente
        angle_reward = 1.0 - abs(theta) / self.theta_threshold  # [0, 1]
        position_reward = 1.0 - abs(x) / self.x_threshold  # [0, 1]
        velocity_penalty = -0.1 * (abs(x_dot) + abs(theta_dot))  # Kazniti velike brzine
        
        # Bonus za malo odstupanje od centra
        stability_bonus = 0.5 if abs(theta) < 0.05 else 0.0
        
        total_reward = angle_reward + 0.3 * position_reward + velocity_penalty + stability_bonus
        
        return total_reward


class StateDiscretizer:
    """
    Poboljšana diskretizacija sa fokusiranjem na kritične regione.
    """
    
    def __init__(self, state_bounds, n_bins_per_dim=15):
        self.state_bounds = state_bounds
        self.n_bins_per_dim = n_bins_per_dim
        
        # Adaptivna diskretizacija - više binova oko centra
        self.bins = []
        for i, (low, high) in enumerate(state_bounds):
            if i == 2:  # Theta - više pažnje oko centra
                # Nelinearno binovanje za theta
                center_bins = np.linspace(-0.1, 0.1, n_bins_per_dim // 2)
                outer_bins_neg = np.linspace(low, -0.1, n_bins_per_dim // 4)
                outer_bins_pos = np.linspace(0.1, high, n_bins_per_dim // 4)
                bins = np.concatenate([outer_bins_neg, center_bins, outer_bins_pos])
            else:
                bins = np.linspace(low, high, n_bins_per_dim)
            
            self.bins.append(np.unique(bins))  # Ukloni duplikate
    
    def discretize(self, state):
        """Diskretizuje kontinuirano stanje."""
        discrete_state = []
        for i, value in enumerate(state):
            value = np.clip(value, self.state_bounds[i][0], self.state_bounds[i][1])
            bin_index = np.digitize(value, self.bins[i]) - 1
            bin_index = np.clip(bin_index, 0, len(self.bins[i]) - 1)
            discrete_state.append(bin_index)
        return tuple(discrete_state)


class OptimizedQLearning:
    """
    Poboljšana implementacija Q-Learning algoritma.
    """
    
    def __init__(self, state_space_size, action_space_size, 
                 learning_rate=0.5, discount_factor=0.99, 
                 epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.02,
                 use_double_q=True):
        
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.use_double_q = use_double_q
        
        # Q-tabele
        self.q_table = {}
        if use_double_q:
            self.q_table_2 = {}
        
        # Statistike
        self.visit_counts = {}
        self.update_counts = {}
        
    def get_q_value(self, state, action, table=1):
        """Dobija Q-vrednost za stanje-akciju par."""
        q_tab = self.q_table if table == 1 else self.q_table_2
        if state not in q_tab:
            q_tab[state] = np.zeros(self.action_space_size)
        return q_tab[state][action]
    
    def set_q_value(self, state, action, value, table=1):
        """Postavlja Q-vrednost za stanje-akciju par."""
        q_tab = self.q_table if table == 1 else self.q_table_2
        if state not in q_tab:
            q_tab[state] = np.zeros(self.action_space_size)
        q_tab[state][action] = value
    
    def choose_action(self, state):
        """Poboljšana epsilon-greedy strategija."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space_size)
        
        # Kombinovane Q-vrednosti za double Q-learning
        if self.use_double_q:
            if state not in self.q_table:
                self.q_table[state] = np.zeros(self.action_space_size)
            if state not in self.q_table_2:
                self.q_table_2[state] = np.zeros(self.action_space_size)
            
            # Prosek oba Q-table
            combined_q = (self.q_table[state] + self.q_table_2[state]) / 2.0
            return np.argmax(combined_q)
        else:
            if state not in self.q_table:
                self.q_table[state] = np.zeros(self.action_space_size)
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """Double Q-learning update."""
        if state not in self.visit_counts:
            self.visit_counts[state] = 0
        self.visit_counts[state] += 1
        
        # Adaptivni learning rate
        adaptive_lr = self.learning_rate / (1 + 0.0001 * self.visit_counts[state])
        
        if self.use_double_q:
            # Randomly choose which Q-table to update
            if np.random.random() < 0.5:
                self._update_q_table(state, action, reward, next_state, done, 1, adaptive_lr)
            else:
                self._update_q_table(state, action, reward, next_state, done, 2, adaptive_lr)
        else:
            self._update_q_table(state, action, reward, next_state, done, 1, adaptive_lr)
    
    def _update_q_table(self, state, action, reward, next_state, done, table, lr):
        """Ažurira određenu Q-tabelu."""
        current_q = self.get_q_value(state, action, table)
        
        if done:
            target = reward
        else:
            if self.use_double_q:
                # Double Q-learning: koristi jednu tabelu za akciju, drugu za vrednost
                other_table = 2 if table == 1 else 1
                
                if next_state not in self.q_table:
                    self.q_table[next_state] = np.zeros(self.action_space_size)
                if next_state not in self.q_table_2:
                    self.q_table_2[next_state] = np.zeros(self.action_space_size)
                
                # Koristi tabelu 'table' za odabir akcije
                q_tab = self.q_table if table == 1 else self.q_table_2
                best_action = np.argmax(q_tab[next_state])
                
                # Koristi drugu tabelu za vrednost
                max_q_next = self.get_q_value(next_state, best_action, other_table)
            else:
                if next_state not in self.q_table:
                    self.q_table[next_state] = np.zeros(self.action_space_size)
                max_q_next = np.max(self.q_table[next_state])
            
            target = reward + self.discount_factor * max_q_next
        
        # Q-learning update
        new_q = current_q + lr * (target - current_q)
        self.set_q_value(state, action, new_q, table)
    
    def decay_epsilon(self):
        """Smanjuje epsilon vrednost."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filename):
        """Čuva trenirani model."""
        data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'visit_counts': self.visit_counts
        }
        if self.use_double_q:
            data['q_table_2'] = self.q_table_2
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    
    def load_model(self, filename):
        """Učitava trenirani model."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.epsilon = data['epsilon']
            self.visit_counts = data['visit_counts']
            if 'q_table_2' in data:
                self.q_table_2 = data['q_table_2']


def train_agent(episodes=3000, render_training=False):
    """
    Poboljšano treniranje agenta.
    """
    print("🚀 Pokretanje poboljšanog treniranja Q-Learning agenta...")
    
    # Inicijalizacija okruženja
    env = CartPoleEnvironment(T=0.01, force_mag=8.0)  # Manje sile i kraći korak
    
    # Poboljšana diskretizacija
    state_bounds = [
        (-2.4, 2.4),    # x pozicija
        (-2.0, 2.0),    # x brzina - smanjeno
        (-0.21, 0.21),  # theta ugao
        (-3.0, 3.0)     # theta brzina - smanjeno
    ]
    discretizer = StateDiscretizer(state_bounds, n_bins_per_dim=12)  # Manje binova
    
    # Poboljšani Q-Learning agent
    agent = OptimizedQLearning(
        state_space_size=12**4,
        action_space_size=2,
        learning_rate=0.3,
        discount_factor=0.99,
        epsilon=0.9,  # Manje početno istraživanje
        epsilon_decay=0.9998,
        epsilon_min=0.05,
        use_double_q=True
    )
    
    # Statistike treniranja
    episode_rewards = []
    episode_lengths = []
    success_rate = deque(maxlen=100)
    best_performance = 0
    
    start_time = time.time()
    
    for episode in range(episodes):
        state = env.reset()
        discrete_state = discretizer.discretize(state)
        
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.choose_action(discrete_state)
            next_state, reward, done = env.step(action)
            next_discrete_state = discretizer.discretize(next_state)
            
            agent.update(discrete_state, action, reward, next_discrete_state, done)
            
            discrete_state = next_discrete_state
            total_reward += reward
            steps += 1
            
            if done or steps >= 1000:
                break
        
        # Statistike
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        success_rate.append(1 if steps >= 200 else 0)  # Snižen prag
        agent.decay_epsilon()
        
        # Praćenje najboljeg performanse
        if steps > best_performance:
            best_performance = steps
            if steps > 500:  # Sačuvaj dobar model
                agent.save_model(f'best_model_{steps}.pkl')
        
        # Progressbar i statistike
        if episode % 200 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            current_success_rate = np.mean(success_rate) if success_rate else 0
            
            print(f"Episode {episode:4d} | "
                  f"Avg Reward: {avg_reward:7.2f} | "
                  f"Avg Length: {avg_length:6.1f} | "
                  f"Success Rate: {current_success_rate:.2%} | "
                  f"Best: {best_performance:4d} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    training_time = time.time() - start_time
    
    print(f"\n✅ Treniranje završeno za {training_time:.2f} sekundi")
    print(f"📊 Finalne statistike:")
    print(f"   - Prosečna nagrada (posledjih 100): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"   - Prosečna dužina epizode: {np.mean(episode_lengths[-100:]):.1f}")
    print(f"   - Najbolja performansa: {best_performance} koraka")
    print(f"   - Stopa uspešnosti: {np.mean(success_rate):.2%}")
    print(f"   - Broj naučenih stanja: {len(agent.q_table)}")
    
    # Čuvanje finalnog modela
    agent.save_model('cart_pole_model.pkl')
    
    return agent, env, discretizer, episode_rewards, episode_lengths


def visualize_training_results(episode_rewards, episode_lengths):
    """Kreira detaljne grafike rezultata treniranja."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Poboljšani rezultati treniranja Q-Learning agenta', fontsize=16, fontweight='bold')
    
    # 1. Nagrada po epizodi
    window = 50
    axes[0, 0].plot(episode_rewards, alpha=0.3, linewidth=0.5, color='blue')
    if len(episode_rewards) > window:
        smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(episode_rewards)), smoothed, 
                        color='red', linewidth=2, label=f'Pokretni prosek ({window})')
    axes[0, 0].set_title('Nagrada po epizodi')
    axes[0, 0].set_xlabel('Epizoda')
    axes[0, 0].set_ylabel('Ukupna nagrada')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Dužina epizode
    axes[0, 1].plot(episode_lengths, alpha=0.3, linewidth=0.5, color='green')
    if len(episode_lengths) > window:
        smoothed = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window-1, len(episode_lengths)), smoothed, 
                        color='red', linewidth=2, label=f'Pokretni prosek ({window})')
    axes[0, 1].set_title('Dužina epizode')
    axes[0, 1].set_xlabel('Epizoda')
    axes[0, 1].set_ylabel('Broj koraka')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Histogram nagrada
    axes[1, 0].hist(episode_rewards, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Distribucija nagrada')
    axes[1, 0].set_xlabel('Nagrada')
    axes[1, 0].set_ylabel('Frekvencija')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Stopa uspešnosti
    success_threshold = 200
    success_rate = [1 if length >= success_threshold else 0 for length in episode_lengths]
    window_size = 100
    if len(success_rate) > window_size:
        success_rate_smooth = np.convolve(success_rate, np.ones(window_size)/window_size, mode='valid')
        axes[1, 1].plot(range(window_size-1, len(success_rate)), success_rate_smooth, linewidth=2)
    axes[1, 1].set_title(f'Stopa uspešnosti (epizode ≥ {success_threshold} koraka)')
    axes[1, 1].set_xlabel('Epizoda')
    axes[1, 1].set_ylabel('Stopa uspešnosti')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def test_agent(agent, env, discretizer, episodes=10):
    """Testira istreniranog agenta."""
    print(f"\n🧪 Testiranje agenta na {episodes} epizoda...")
    
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Bez eksploracije tokom testiranja
    
    test_rewards = []
    test_lengths = []
    
    for episode in range(episodes):
        state = env.reset()
        discrete_state = discretizer.discretize(state)
        
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.choose_action(discrete_state)
            next_state, reward, done = env.step(action)
            next_discrete_state = discretizer.discretize(next_state)
            
            discrete_state = next_discrete_state
            total_reward += reward
            steps += 1
            
            if done or steps >= 1000:
                break
        
        test_rewards.append(total_reward)
        test_lengths.append(steps)
        
        print(f"Test epizoda {episode + 1}: {steps} koraka, nagrada: {total_reward:.2f}")
    
    agent.epsilon = original_epsilon
    
    print(f"\n📈 Rezultati testiranja:")
    print(f"   - Prosečna nagrada: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"   - Prosečna dužina: {np.mean(test_lengths):.1f} ± {np.std(test_lengths):.1f}")
    print(f"   - Najbolji rezultat: {max(test_lengths)} koraka")
    print(f"   - Stopa uspešnosti (≥200): {sum(1 for length in test_lengths if length >= 200) / len(test_lengths):.2%}")
    
    return test_rewards, test_lengths


def animate_cart_pole(env, agent, discretizer, max_steps=500):
    """Kreira animiranu vizualizaciju cart-pole simulacije."""
    print("🎬 Kreiranje animirane simulacije...")
    
    # Resetovanje okruženja
    state = env.reset()
    discrete_state = discretizer.discretize(state)
    
    # Čuvanje podataka za animaciju
    positions = []
    angles = []
    actions = []
    
    agent.epsilon = 0.0  # Bez eksploracije
    
    for step in range(max_steps):
        action = agent.choose_action(discrete_state)
        next_state, reward, done = env.step(action)
        
        positions.append(next_state[0])
        angles.append(next_state[2])
        actions.append(action)
        
        discrete_state = discretizer.discretize(next_state)
        
        if done:
            break
    
    print(f"Simulacija završena nakon {len(positions)} koraka")
    
    # Kreiranje animacije
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Cart-Pole Animirana Simulacija', fontsize=16, fontweight='bold')
    
    # 1. Cart-pole vizualizacija
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-1, 2)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Cart-Pole Pozicija')
    ax1.set_xlabel('X pozicija (m)')
    ax1.set_ylabel('Y pozicija (m)')
    
    # Elementi za animaciju
    cart_patch = plt.Rectangle((0, 0), 0.4, 0.2, fill=True, color='blue', alpha=0.7)
    ax1.add_patch(cart_patch)
    pole_line, = ax1.plot([], [], 'r-', linewidth=6, label='Štap')
    ax1.legend()
    
    # 2. Ugao tokom vremena
    ax2.set_xlim(0, len(angles))
    ax2.set_ylim(-0.25, 0.25)
    ax2.set_title('Ugao štapa')
    ax2.set_xlabel('Vreme (koraci)')
    ax2.set_ylabel('Ugao (rad)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='g', linestyle='--', alpha=0.5, label='Cilj')
    ax2.axhline(y=0.2094, color='r', linestyle='--', alpha=0.5, label='Granica')
    ax2.axhline(y=-0.2094, color='r', linestyle='--', alpha=0.5)
    
    angle_line, = ax2.plot([], [], 'b-', linewidth=2)
    current_angle_point, = ax2.plot([], [], 'ro', markersize=8)
    ax2.legend()
    
    # 3. Pozicija tokom vremena
    ax3.set_xlim(0, len(positions))
    ax3.set_ylim(-3, 3)
    ax3.set_title('Pozicija kolica')
    ax3.set_xlabel('Vreme (koraci)')
    ax3.set_ylabel('Pozicija (m)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='g', linestyle='--', alpha=0.5, label='Centar')
    ax3.axhline(y=2.4, color='r', linestyle='--', alpha=0.5, label='Granica')
    ax3.axhline(y=-2.4, color='r', linestyle='--', alpha=0.5)
    
    position_line, = ax3.plot([], [], 'g-', linewidth=2)
    current_position_point, = ax3.plot([], [], 'go', markersize=8)
    ax3.legend()
    
    # 4. Akcije tokom vremena
    ax4.set_xlim(0, len(actions))
    ax4.set_ylim(-0.5, 1.5)
    ax4.set_title('Akcije agenta')
    ax4.set_xlabel('Vreme (koraci)')
    ax4.set_ylabel('Akcija (0=Levo, 1=Desno)')
    ax4.grid(True, alpha=0.3)
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Levo', 'Desno'])
    
    action_line, = ax4.plot([], [], 'purple', linewidth=2, marker='o', markersize=4)
    current_action_point, = ax4.plot([], [], 'ro', markersize=8)
    
    # Tekst za prikaz trenutnih vrednosti
    info_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def animate(frame):
        if frame >= len(positions):
            return
        
        # Trenutne vrednosti
        x_pos = positions[frame]
        theta = angles[frame]
        current_action = actions[frame]
        
        # Pozicija kolica
        cart_x = x_pos - 0.2
        cart_y = -0.1
        cart_patch.set_x(cart_x)
        cart_patch.set_y(cart_y)
        # Pozicija kolica
        cart_x = x_pos - 0.2
        cart_y = -0.1
        cart_patch.set_x(cart_x)
        cart_patch.set_y(cart_y)
        
        # Pozicija štapa
        pole_x = x_pos
        pole_y = 0
        pole_tip_x = pole_x + env.l * np.sin(theta)
        pole_tip_y = pole_y + env.l * np.cos(theta)
        
        pole_line.set_data([pole_x, pole_tip_x], [pole_y, pole_tip_y])
        
        # Ažuriranje grafika
        angle_line.set_data(range(frame + 1), angles[:frame + 1])
        current_angle_point.set_data([frame], [theta])
        
        position_line.set_data(range(frame + 1), positions[:frame + 1])
        current_position_point.set_data([frame], [x_pos])
        
        action_line.set_data(range(frame + 1), actions[:frame + 1])
        current_action_point.set_data([frame], [current_action])
        
        # Ažuriranje info teksta
        info_text.set_text(f'Korak: {frame + 1}\n'
                          f'Pozicija: {x_pos:.3f} m\n'
                          f'Ugao: {theta:.3f} rad ({np.degrees(theta):.1f}°)\n'
                          f'Akcija: {"Desno" if current_action == 1 else "Levo"}')
        
        return (cart_patch, pole_line, angle_line, current_angle_point, 
                position_line, current_position_point, action_line, 
                current_action_point, info_text)
    
    # Kreiranje animacije
    anim = FuncAnimation(fig, animate, frames=len(positions), 
                        interval=50, blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    # Čuvanje animacije (opciono)
    try:
        anim.save('cart_pole_animation.gif', writer='pillow', fps=20)
        print("💾 Animacija sačuvana kao 'cart_pole_animation.gif'")
    except Exception as e:
        print(f"⚠️  Greška pri čuvanju animacije: {e}")
    
    return anim


def main():
    """Glavna funkcija za pokretanje treniranja i testiranja."""
    print("🎯 Cart-Pole Q-Learning Simulacija")
    print("=" * 50)
    
    # Treniranje agenta
    agent, env, discretizer, episode_rewards, episode_lengths = train_agent(
        episodes=3000, render_training=False
    )
    
    # Vizualizacija rezultata treniranja
    visualize_training_results(episode_rewards, episode_lengths)
    
    # Testiranje agenta
    test_rewards, test_lengths = test_agent(agent, env, discretizer, episodes=10)
    
    # Animirana simulacija
    print("\n🎬 Pokretanje animirane simulacije...")
    animate_cart_pole(env, agent, discretizer, max_steps=500)
    
    # Analiza naučenih strategija
    print("\n📊 Analiza naučenih strategija:")
    print(f"   - Ukupno naučenih stanja: {len(agent.q_table)}")
    print(f"   - Prosečna Q-vrednost: {np.mean([np.mean(q_vals) for q_vals in agent.q_table.values()]):.3f}")
    
    # Najčešće posećena stanja
    if agent.visit_counts:
        most_visited = sorted(agent.visit_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"   - Najčešće posećena stanja:")
        for i, (state, count) in enumerate(most_visited):
            print(f"     {i+1}. Stanje {state}: {count} poseta")
    
    print("\n✅ Simulacija završena!")


if __name__ == "__main__":
    main()
