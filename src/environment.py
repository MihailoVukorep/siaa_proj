import numpy as np


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
        
        #! Ugaona akceleracija θ̈ (ugaono ubrzanje)
        numerator_theta = self.g * sin_theta - cos_theta * temp
        denominator_theta = self.l * (4.0/3.0 - self.m * cos_theta**2 / self.total_mass)
        
        # Dodavanje epsilon za numeričku stabilnost
        denominator_theta = np.clip(denominator_theta, 1e-6, None)
        f_theta = numerator_theta / denominator_theta
        
        #! Linearna akceleracija ẍ (translaciono-linearno ubrzanje)
        f_x = temp - self.m * self.l * f_theta * cos_theta / self.total_mass
        
        return f_x, f_theta
    
    def step(self, action):
        """Poboljšana implementacija koraka."""
        # Konvertovanje diskretne akcije u silu
        force = self.force_mag if action == 1 else -self.force_mag
        
        #! Trenutno stanje (x-pozicija kolica, x_dot-brzina, theta-ugao, theta_dot-ugaona brzina)
        x, x_dot, theta, theta_dot = self.state
        
        # Stabilizacija sistema - manje steps za integraciju
        for _ in range(2):  # Podeli korak na pola
            # Računanje izvoda
            f_x, f_theta = self._get_derivatives(self.state, force)
            
            # Euler diskretizacija sa pola koraka
            half_T = self.T / 2
            
            #? računanje novih stanja
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
