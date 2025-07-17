import numpy as np


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
