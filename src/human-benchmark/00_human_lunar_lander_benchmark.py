"""
Human Lunar Lander Interactive Benchmark (edreate.com)

Allows a human to manually control the Gymnasium LunarLander-v3 environment.

Dependencies:
  - gymnasium[box2d] (pip install gymnasium[box2d])
  - pygame         (pip install pygame)

Usage:
  python 00_human_lunar_lander_benchmark.py
  Enter number of episodes when prompted.

Controls (hold for continuous thrust):
  - Left arrow : fire left engine (action 1)
  - Up arrow   : fire main engine (action 2)
  - Right arrow: fire right engine (action 3)
  - No key     : do nothing (action 0)
  - ESC or close window: quit early

License:
  MIT License
  Copyright (c) 2025 edreate.com
"""

import gymnasium as gym
import pygame


def run_human_lunar_lander(
    episodes: int = 5,
    screen_width: int = 800,
    screen_height: int = 600,
    fps: int = 30,
):
    """
    Runs the LunarLander-v3 environment with human keyboard controls.

    On-screen you’ll see:
      • Semi-transparent top bar with:
        - Episode progress   (Ep x/y)
        - Step count         (Step: z)
        - Current action     (Action: None/Left/Main/Right)
        - FPS                (real-time frame rate)
      • Big, centered total reward (Reward: xxx.x)
    Tracks reward and steps per episode; prints summary at the end.
    """
    # prepare environment and pygame
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    pygame.init()
    pygame.display.set_caption("Human Lunar Lander")
    clock = pygame.time.Clock()

    all_rewards, all_steps = [], []

    action_names = {0: "None", 1: "Left", 2: "Main", 3: "Right"}

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False
        cumulative_reward = 0.0
        steps = 0

        # create window and fonts once per episode
        screen = pygame.display.set_mode((screen_width, screen_height))
        small_font = pygame.font.Font(None, 24)
        large_font = pygame.font.Font(None, 36)

        while not done:
            # 1) choose action from keys
            action = 0
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action = 1
            elif keys[pygame.K_UP]:
                action = 2
            elif keys[pygame.K_RIGHT]:
                action = 3

            # 2) handle quit
            for evt in pygame.event.get():
                if evt.type == pygame.QUIT or (
                    evt.type == pygame.KEYDOWN and evt.key == pygame.K_ESCAPE
                ):
                    done = True

            # 3) step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            cumulative_reward += reward
            steps += 1
            done = done or terminated or truncated

            # 4) render frame to pygame
            frame = env.render()
            surf = pygame.surfarray.make_surface(frame.transpose((1, 0, 2)))
            screen.blit(
                pygame.transform.scale(surf, (screen_width, screen_height)), (0, 0)
            )

            # 5) draw semi-transparent top bar
            bar_height = large_font.get_height() + small_font.get_height() + 20
            bar = pygame.Surface((screen_width, bar_height), pygame.SRCALPHA)
            bar.fill((0, 0, 0, 160))  # black with alpha
            screen.blit(bar, (0, 0))

            # 6) overlay text
            # Episode and step
            screen.blit(
                small_font.render(f"Ep {ep}/{episodes}", True, (255, 255, 255)),
                (10, 10),
            )
            screen.blit(
                small_font.render(f"Step: {steps}", True, (255, 255, 255)),
                (10, 10 + small_font.get_height()),
            )
            # Action
            screen.blit(
                small_font.render(
                    f"Action: {action_names[action]}", True, (255, 255, 255)
                ),
                (150, 10),
            )
            # FPS
            fps_text = f"FPS: {int(clock.get_fps())}"
            txt_surf = small_font.render(fps_text, True, (255, 255, 255))
            screen.blit(txt_surf, (screen_width - txt_surf.get_width() - 10, 10))

            # Big centered cumulative reward
            reward_surf = large_font.render(
                f"Reward: {cumulative_reward:.1f}", True, (255, 215, 0)
            )
            reward_rect = reward_surf.get_rect(
                center=(screen_width // 2, bar_height // 2)
            )
            screen.blit(reward_surf, reward_rect)

            # 7) flip & tick
            pygame.display.flip()
            clock.tick(fps)

        all_rewards.append(cumulative_reward)
        all_steps.append(steps)
        print(f"Episode {ep} finished - Reward: {cumulative_reward:.2f}, Steps: {steps}")

    env.close()
    pygame.quit()

    # Final summary
    if all_rewards:
        print("\n--- Human Benchmark Results ---")
        print(f"Episodes played    : {episodes}")
        print(f"Max Reward         : {max(all_rewards):.2f}")
        print(f"Average Reward     : {sum(all_rewards) / len(all_rewards):.2f}")
        print(f"Worst Reward       : {min(all_rewards):.2f}")
        print(f"Average Steps/Ep   : {sum(all_steps) / len(all_steps):.1f}")


if __name__ == "__main__":
    try:
        n_episodes = int(input("How many episodes would you like to try? "))
    except ValueError:
        print("Invalid input. Defaulting to 5 episodes.")
        n_episodes = 5
    run_human_lunar_lander(episodes=n_episodes)
