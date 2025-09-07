import numpy as np
import pickle


class OptimizedQLearning:
    """
    Poboljšana implementacija Q-Learning algoritma.
    """
    
    def __init__(self, state_space_size, action_space_size, 
                 learning_rate=0.5, discount_factor=0.99, 
                 epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.02,
                 use_double_q=True):
        
        self.state_space_size = state_space_size # Broj diskretnih stanja, 12^n za n dimenzija
        self.action_space_size = action_space_size # Broj mogućih akcija (levo/desno)
        self.learning_rate = learning_rate # Koliko brzo Q-vrednosti reaguje na novo iskustvo
        self.discount_factor = discount_factor # Koliko agent vrednuje buduće nagrade
        self.epsilon = epsilon # Početna vrednost epsilon za epsilon-greedy
        self.epsilon_decay = epsilon_decay # Koliko se epsilon smanjuje nakon svake epizode
        self.epsilon_min = epsilon_min # Minimalna vrednost epsilon, ispod koje se ne smanjuje
        self.use_double_q = use_double_q # Da li koristi Double Q-Learning
        
        # Q-tabele
        self.q_table = {}
        if use_double_q:
            self.q_table_2 = {}
        
        # Statistike
        self.visit_counts = {} # Broj poseta svakom stanju
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
