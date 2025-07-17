import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


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


def animate_cart_pole(env, agent, discretizer, max_steps=500):
    """Kreira animiranu vizualizaciju cart-pole simulacije."""
    print("Kreiranje animirane simulacije...")
    
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
        print("Animacija sačuvana kao 'cart_pole_animation.gif'")
    except Exception as e:
        print(f"Greška pri čuvanju animacije: {e}")
    
    return anim
