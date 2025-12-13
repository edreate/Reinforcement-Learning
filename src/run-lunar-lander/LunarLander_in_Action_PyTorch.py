import torch
import gymnasium as gym
from pathlib import Path
from typing import Type
from Visualizer import PygameVisualizer
import torch.nn as nn
from SAC import ActorNetwork


def load_model(
    filename: Path,
    model_class: Type[nn.Module],
    n_observations: int,
    n_actions: int,
) -> nn.Module:
    """
    Loads a model of the specified class with given parameters.

    Args:
        filename (Path): Path to the saved model file.
        model_class (Type[nn.Module]): The class of the model to load.
        n_observations (int): Number of observations (input features).
        n_actions (int): Number of actions (output features).

    Returns:
        nn.Module: Loaded model instance.
    """
    try:
        model = model_class(
            state_dim=n_observations, action_dim=n_actions, hidden_dim=256
        )
        model.load_state_dict(torch.load(filename, weights_only=True))
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")


def run_and_control_lunar_lander(
    actor: nn.Module,
    render_fps: int = 20,
    continuous: bool = False,
):
    """
    Runs the Lunar Lander simulation and controls the landing using the provided trained model.

    Args:
        actor (nn.Module): Trained model for action selection.
        render_fps (int): Frames per second for rendering.
    """
    env = gym.make("LunarLander-v3", render_mode="rgb_array", continuous=continuous)

    visualizer = PygameVisualizer(
        caption=f"Lunar Lander Control in {'continuous' if continuous is True else 'discrete'}",
        text_color=(200, 200, 200),
    )
    visualizer.initialize()

    state, _ = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    done = False
    time_step: int = 0
    cumulative_reward: float = 0
    with torch.no_grad():
        while not done:
            action = actor.get_action(state_tensor)
            observation, reward, terminated, truncated, _ = env.step(action)
            state_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

            cumulative_reward += float(reward)
            time_step += 1

            frame = env.render()
            visualizer.render_frame(frame)

            info_text = (
                f"Time step: {time_step}      Reward: {cumulative_reward:.2f}     "
                f"Lander X: {observation[0]:.2f}       Lander Y: {observation[1]:.2f}     "
                f"Velocity X: {observation[2]:.2f}      Velocity Y: {observation[3]:.2f}"
            )
            visualizer.overlay_info(info_text)
            visualizer.update_display()

            done = terminated or truncated
            visualizer.tick(render_fps)

    env.close()
    visualizer.quit()


if __name__ == "__main__":
    MODEL_FILE_PATH = Path(
        "output/sac_continuous_lunar_lander_training_2024-12-23_12-52-16/policy_network.pth"
    )
    MODEL_CLASS = ActorNetwork
    CONTINUOUS = True

    # Verbose run information
    print(f"Using model: {MODEL_FILE_PATH}")
    print(f"Model class: {MODEL_CLASS.__name__}")
    print(f"Action space type: {'Continuous' if CONTINUOUS else 'Discrete'}")

    # Number of actions based on Environment Setting
    n_actions = 4
    if CONTINUOUS:
        n_actions = 2

    # Initialize the environment to get observation and action space sizes
    env = gym.make("LunarLander-v3")
    n_observations = int(env.observation_space.shape[0])
    env.close()

    # Load Agent model
    agent = load_model(MODEL_FILE_PATH, MODEL_CLASS, n_observations, n_actions)

    # Run and Control Lunar Lander
    run_and_control_lunar_lander(agent, continuous=CONTINUOUS)
