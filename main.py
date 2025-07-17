import os
import numpy as np

from src.environment import CartPoleEnvironment
from src.discretizer import StateDiscretizer
from src.q_learning import OptimizedQLearning
from src.training import train_agent, test_agent
from src.visualization import visualize_training_results, animate_cart_pole

os.makedirs("result", exist_ok=True)

def main():
    """Glavna funkcija za pokretanje treniranja i testiranja."""
    print("Cart-Pole Q-Learning Simulacija")
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
    print("\nPokretanje animirane simulacije...")
    animate_cart_pole(env, agent, discretizer, max_steps=500)
    
    # Analiza naučenih strategija
    print("\nAnaliza naučenih strategija:")
    print(f"   - Ukupno naučenih stanja: {len(agent.q_table)}")
    print(f"   - Prosečna Q-vrednost: {np.mean([np.mean(q_vals) for q_vals in agent.q_table.values()]):.3f}")
    
    # Najčešće posećena stanja
    if agent.visit_counts:
        most_visited = sorted(agent.visit_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"   - Najčešće posećena stanja:")
        for i, (state, count) in enumerate(most_visited):
            print(f"     {i+1}. Stanje {state}: {count} poseta")
    
    print("\nSimulacija završena!")


if __name__ == "__main__":
    main()
