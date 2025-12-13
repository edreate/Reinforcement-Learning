from pathlib import Path
from typing import Optional

import numpy as np
import pygame
import gymnasium as gym
import onnxruntime as ort

# ------------------------------
# ONNX utilities
# ------------------------------


def load_onnx_session(
    filename: Path, providers: Optional[list[str]] = None
) -> ort.InferenceSession:
    """
    Load an ONNX model for inference.
    """
    if providers is None:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(filename.as_posix(), providers=providers)


def onnx_policy_action(
    session: ort.InferenceSession,
    state: np.ndarray,
    *,
    input_name: Optional[str] = None,
    output_name: Optional[str] = None,
    continuous: bool = False,
    apply_tanh: bool = False,
) -> np.ndarray | int:
    """
    Compute an action from an ONNX policy given a single environment state.

    - Assumes input shape [batch, obs_dim]; adds batch dim automatically.
    - For discrete: output is Q-values/logits -> argmax -> int action.
    - For continuous: output is action vector; optionally apply tanh.
    """
    if input_name is None:
        input_name = session.get_inputs()[0].name
    if output_name is None:
        output_name = session.get_outputs()[0].name

    x = np.asarray(state, dtype=np.float32)[None, ...]  # [1, obs_dim]
    out = session.run([output_name], {input_name: x})[0]
    out = np.asarray(out)

    if not continuous:
        return int(out[0].argmax(-1))

    action = out[0]
    if apply_tanh:
        action = np.tanh(action)
    return action


# ------------------------------
# Pygame HUD + control loop
# ------------------------------

DISCRETE_ACTION_NAMES = {
    0: "noop",
    1: "left engine",
    2: "main engine",
    3: "right engine",
}


def _format_obs(obs: np.ndarray, continuous: bool) -> list[str]:
    # LunarLander obs (discrete & continuous): 8 values
    # [x, y, vx, vy, angle, ang_vel, left_contact, right_contact]
    names = [
        ("Lander X", obs[0]),
        ("Lander Y", obs[1]),
        ("Vel X", obs[2]),
        ("Vel Y", obs[3]),
        ("Angle", obs[4]),
        ("Ang Vel", obs[5]),
        ("Left Contact", bool(obs[6] >= 0.5)),
        ("Right Contact", bool(obs[7] >= 0.5)),
    ]
    lines: list[str] = []
    for k, v in names:
        if isinstance(v, bool):
            lines.append(f"{k:<14}: {str(v):>5}")
        else:
            lines.append(f"{k:<14}: {v:>8.3f}")
    lines.append(f"Mode           : {'continuous' if continuous else 'discrete'}")
    return lines


def _draw_hud(
    surface: pygame.Surface,
    panel_rect: pygame.Rect,
    lines: list[str],
    *,
    title: str,
    extras: list[str],
) -> None:
    # Panel background
    overlay = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 140))  # translucent
    surface.blit(overlay, panel_rect.topleft)

    # Text
    font_title = pygame.font.SysFont(None, 26, bold=True)
    font_text = pygame.font.SysFont(None, 22)
    y = panel_rect.top + 10
    surface.blit(
        font_title.render(title, True, (230, 230, 230)), (panel_rect.left + 12, y)
    )
    y += 16

    # Stats lines
    for s in lines:
        y += 20
        surface.blit(
            font_text.render(s, True, (220, 220, 220)), (panel_rect.left + 12, y)
        )

    # Divider
    y += 16
    pygame.draw.line(
        surface,
        (200, 200, 200),
        (panel_rect.left + 10, y),
        (panel_rect.right - 10, y),
        1,
    )

    # Extra lines
    for s in extras:
        y += 20
        surface.blit(
            font_text.render(s, True, (220, 220, 220)), (panel_rect.left + 12, y)
        )


def _draw_final_banner(
    screen: pygame.Surface, text_lines: list[str], *, success: bool
) -> None:
    W, H = screen.get_size()
    overlay = pygame.Surface((W, H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))  # darken the whole window
    screen.blit(overlay, (0, 0))

    color = (40, 200, 90) if success else (220, 70, 70)
    title_font = pygame.font.SysFont(None, 64, bold=True)
    body_font = pygame.font.SysFont(None, 30)

    # Title
    title = "SUCCESSFUL LANDING" if success else "EPISODE FAILED"
    title_surface = title_font.render(title, True, color)
    title_rect = title_surface.get_rect(center=(W // 2, H // 2 - 60))
    screen.blit(title_surface, title_rect)

    # Body lines
    y = H // 2
    for line in text_lines:
        surf = body_font.render(line, True, (235, 235, 235))
        rect = surf.get_rect(center=(W // 2, y))
        screen.blit(surf, rect)
        y += 36

    pygame.display.flip()


def run_and_control_lunar_lander(
    session: ort.InferenceSession,
    *,
    render_fps: int = 20,
    continuous: bool = False,
    apply_tanh: bool = False,
):
    """
    Runs LunarLander-v3 with an ONNX policy and shows a rich HUD with all stats.
    """
    # Choose renderer
    env = gym.make("LunarLander-v3", render_mode="rgb_array", continuous=continuous)

    state, _ = env.reset()
    done = False
    t = 0
    cum_reward = 0.0
    last_step_reward = 0.0
    last_action: Optional[np.ndarray | int] = None
    last_terminated = False
    last_truncated = False

    # First frame to size window
    first_frame = env.render()  # H x W x 3
    fh, fw = first_frame.shape[:2]

    # HUD width (right-side panel)
    HUD_W = 360
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((fw + HUD_W, fh))
    pygame.display.set_caption(
        f"Lunar Lander (ONNX) - {'Continuous' if continuous else 'Discrete'}"
    )
    clock = pygame.time.Clock()

    def show(frame_np: np.ndarray):
        # Pump events so the OS actually shows/refreshes the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                raise SystemExit

        # Left: game frame
        surf = pygame.surfarray.make_surface(frame_np.swapaxes(0, 1))  # W x H
        screen.blit(surf, (0, 0))

        # Right: HUD panel
        panel_rect = pygame.Rect(fw, 0, HUD_W, fh)

        # Build lines
        obs_lines = _format_obs(state, continuous)

        # Action description
        if last_action is None:
            action_str = "N/A"
        else:
            if continuous:
                # Continuous: action is a vector, commonly 2 floats: [main_engine, lateral]
                action_arr = np.asarray(last_action, dtype=float).flatten()
                action_str = "[" + ", ".join(f"{v: .3f}" for v in action_arr) + "]"
            else:
                name = DISCRETE_ACTION_NAMES.get(int(last_action), "?")
                action_str = f"{int(last_action)} ({name})"

        extras = [
            f"Time step      : {t}",
            f"Step reward    : {last_step_reward: .3f}",
            f"Cumulative R   : {cum_reward: .3f}",
            f"Action taken   : {action_str}",
            f"Terminated?    : {str(last_terminated)}",
            f"Truncated?     : {str(last_truncated)}",
            f"Target FPS     : {render_fps}",
            f"Real FPS       : {clock.get_fps():6.2f}",
        ]

        _draw_hud(
            screen,
            panel_rect,
            obs_lines,
            title="Lunar Lander - Live Stats",
            extras=extras,
        )
        pygame.display.flip()

    # Initial draw
    show(first_frame)

    while not done:
        # Compute action from ONNX
        action = onnx_policy_action(
            session, state, continuous=continuous, apply_tanh=apply_tanh
        )
        observation, reward, terminated, truncated, _ = env.step(action)

        # Update trackers
        state = observation
        last_action = action
        last_step_reward = float(reward)
        cum_reward += last_step_reward
        t += 1
        last_terminated = bool(terminated)
        last_truncated = bool(truncated)
        done = last_terminated or last_truncated

        # Draw
        frame = env.render()
        show(frame)

        clock.tick(render_fps)

    # Prefer env.unwrapped.game_over (True on crash). If missing, default False.
    game_over = bool(getattr(env.unwrapped, "game_over", False))
    landed_ok = last_terminated and not last_truncated and (not game_over)
    solved_score = cum_reward >= 200.0
    success = landed_ok or solved_score

    summary_lines = [
        f"Total steps: {t}",
        f"Cumulative reward: {cum_reward: .2f}",
        f"Landed OK: {landed_ok}",
        f"Solved threshold (>=200): {solved_score}",
    ]

    # Draw one more time so game frame + HUD are visible, then overlay banner
    frame = env.render()
    show(frame)
    _draw_final_banner(screen, summary_lines, success=success)
    pygame.time.delay(1800)  # keep the banner for ~1.8s

    # Also log to stdout
    print("\n=== Episode Summary ===")
    print(f"Steps: {t}")
    print(f"Cumulative reward: {cum_reward:.2f}")
    print(f"Landed OK (terminated & not crashed): {landed_ok}")
    print(f"Solved threshold (>= 200): {solved_score}")
    print(f"SUCCESS: {success}")

    env.close()
    pygame.quit()


if __name__ == "__main__":
    # Point this to your ONNX file
    MODEL_FILE_PATH = Path(
        "training_output_lunar_lander/discrete/deep_q_network_lunar_lander/trained_dqn_2025-12-13_23-18.onnx"
    )

    # Discrete agent -> False, Continuous agent -> True
    CONTINUOUS = False

    # If your continuous policy outputs unbounded values, set this True to squash to [-1, 1]
    APPLY_TANH = False

    print(f"Using ONNX model: {MODEL_FILE_PATH}")
    print(f"Action space type: {'Continuous' if CONTINUOUS else 'Discrete'}")

    session = load_onnx_session(MODEL_FILE_PATH)
    run_and_control_lunar_lander(
        session, continuous=CONTINUOUS, apply_tanh=APPLY_TANH, render_fps=20
    )
