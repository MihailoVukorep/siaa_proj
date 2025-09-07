#!/usr/bin/env python

import argparse
import os
import sys
from src.training import train_agent, test_agent
from src.environment import CartPoleEnvironment
from src.discretizer import StateDiscretizer
from src.q_learning import OptimizedQLearning


def create_agent_and_env():
    """Creates and returns agent, environment, and discretizer instances."""
    # Initialize environment
    env = CartPoleEnvironment(T=0.01, force_mag=8.0)
    
    # Initialize discretizer
    state_bounds = [
        (-2.4, 2.4),    # x pozicija
        (-2.0, 2.0),    # x brzina
        (-0.21, 0.21),  # theta ugao
        (-3.0, 3.0)     # theta brzina
    ]
    discretizer = StateDiscretizer(state_bounds, n_bins_per_dim=12)
    
    # Initialize agent
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
    
    return agent, env, discretizer


def main():
    parser = argparse.ArgumentParser(description='Cart-Pole Q-Learning Agent')
    
    # Main mode arguments
    parser.add_argument('--train', action='store_true', 
                       help='Train the agent')
    parser.add_argument('--test', action='store_true',
                       help='Test the agent')
    parser.add_argument('--load', type=str, default=None,
                       help='Load model from specified path (e.g., --load result/cart_pole_model.pkl)')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=3000,
                       help='Number of training episodes (default: 3000)')
    parser.add_argument('--save-path', type=str, default='result/cart_pole_model.pkl',
                       help='Path to save the trained model (default: result/cart_pole_model.pkl)')
    
    # Testing parameters  
    parser.add_argument('--test-episodes', type=int, default=10,
                       help='Number of test episodes (default: 10)')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment during testing')
    
    args = parser.parse_args()
    
    # Create result directory if it doesn't exist
    os.makedirs('result', exist_ok=True)
    
    # Create agent, environment, and discretizer
    agent, env, discretizer = create_agent_and_env()
    
    # Handle load model
    if args.load:
        if os.path.exists(args.load):
            print(f"Loading model from: {args.load}")
            agent.load_model(args.load)
            print("Model loaded successfully!")
        else:
            print(f"Error: Model file '{args.load}' not found!")
            sys.exit(1)
    
    # Handle training
    if args.train:
        if args.load:
            print("Continuing training from loaded model...")
        else:
            print("Starting training from scratch...")
            
        agent, env, discretizer, episode_rewards, episode_lengths = train_agent(
            episodes=args.episodes,
            agent=agent,
            env=env, 
            discretizer=discretizer
        )
        
        # Save the trained model
        print(f"Saving model to: {args.save_path}")
        agent.save_model(args.save_path)
        print("Model saved successfully!")
    
    # Handle testing
    if args.test:
        if not args.load and not args.train:
            print("Error: No model loaded! Use --load to load a model or --train to train first.")
            sys.exit(1)
            
        print("Testing the agent...")
        test_rewards, test_lengths = test_agent(
            agent=agent,
            env=env,
            discretizer=discretizer,
            episodes=args.test_episodes
        )
    
    # If no action specified, show help
    if not (args.train or args.test):
        parser.print_help()


if __name__ == "__main__":
    main()
