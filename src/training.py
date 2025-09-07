import numpy as np
import time
import os
import matplotlib.pyplot as plt
from collections import deque

from .environment import CartPoleEnvironment
from .discretizer import StateDiscretizer
from .q_learning import OptimizedQLearning
from .visualization import visualize_training_results, animate_cart_pole


def train_agent(episodes=3000, render=False, agent=None, env=None, discretizer=None):
    """
    Poboljšano treniranje agenta sa opcijom za prenos komponenti.
    """
    print("Pokretanje poboljšanog treniranja Q-Learning agenta...")
    
    # Ako komponente nisu prosleđene, kreiraj ih
    if env is None:
        env = CartPoleEnvironment(T=0.01, force_mag=8.0)
    
    if discretizer is None:
        state_bounds = [
            (-2.4, 2.4),    # x pozicija
            (-2.0, 2.0),    # x brzina - smanjeno
            (-0.21, 0.21),  # theta ugao
            (-3.0, 3.0)     # theta brzina - smanjeno
        ]
        discretizer = StateDiscretizer(state_bounds, n_bins_per_dim=12)
    
    if agent is None:
        agent = OptimizedQLearning(
            state_space_size=12**4,
            action_space_size=2,
            learning_rate=0.3,
            discount_factor=0.99,
            epsilon=0.9,
            epsilon_decay=0.9998,
            epsilon_min=0.05,
            use_double_q=True
        )
    
    # Print current agent status
    print(f"Starting training with:")
    print(f"   - Epsilon: {agent.epsilon:.3f}")
    print(f"   - Learned states: {len(agent.q_table)}")
    if hasattr(agent, 'q_table_2'):
        print(f"   - Double Q-Learning enabled")
    
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
        success_rate.append(1 if steps >= 200 else 0)
        agent.decay_epsilon()
        
        # Praćenje najbolje performance
        if steps > best_performance:
            best_performance = steps
            if steps > 500:  # Sačuvaj dobar model
                os.makedirs("result", exist_ok=True)
                agent.save_model(os.path.join("result", f'best_model_{steps}.pkl'))
        
        # Progress bar i statistike
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
    
    print(f"\nTreniranje završeno za {training_time:.2f} sekundi")
    print(f"Finalne statistike:")
    print(f"   - Prosečna nagrada (poslednjih 100): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"   - Prosečna dužina epizode: {np.mean(episode_lengths[-100:]):.1f}")
    print(f"   - Najbolja performansa: {best_performance} koraka")
    print(f"   - Stopa uspešnosti: {np.mean(success_rate):.2%}")
    print(f"   - Broj naučenih stanja: {len(agent.q_table)}")

    if render:
        # Vizualizacija rezultata treniranja
        visualize_training_results(episode_rewards, episode_lengths)
    
    return agent, env, discretizer, episode_rewards, episode_lengths


def test_agent(agent, env, discretizer, episodes=10, render=False):
    """Testira istreniranog agenta."""
    print(f"\nTestiranje agenta na {episodes} epizoda...")
    
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
    
    print(f"\nRezultati testiranja:")
    print(f"   - Prosečna nagrada: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"   - Prosečna dužina: {np.mean(test_lengths):.1f} ± {np.std(test_lengths):.1f}")
    print(f"   - Najbolji rezultat: {max(test_lengths)} koraka")
    print(f"   - Stopa uspešnosti (≥200): {sum(1 for length in test_lengths if length >= 200) / len(test_lengths):.2%}")

    if render:
        # Animirana simulacija
        print("\nPokretanje animirane simulacije...")
        try:
            animate_cart_pole(env, agent, discretizer, max_steps=500)
        except KeyboardInterrupt:
            print("Simulacija prekinuta od strane korisnika.")
        except Exception as e:
            print(f"Greška tokom animacije: {e}")
        finally:
            # Osiguravamo da se svi matplotlib resursi zatvaraju
            plt.close('all')
        
        # Animirana simulacija
        print("\nPokretanje animirane simulacije...")
        animate_cart_pole(env, agent, discretizer, max_steps=500)
    
    return test_rewards, test_lengths
