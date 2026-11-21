"""The actor-critic network, used unchanged by pretraining and by PPO.

``forward`` gives the policy's logits (what the exported Seddler bot runs), ``value`` the
state's value, ``action_and_value`` what PPO needs. Inputs are standardized with a running
mean and spread kept in the network, so the exported bot does the same.
"""

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from seddler_config import ModelConfig

MASKED_LOGIT = -1e9  # the logit of an illegal action: its probability is exactly 0
MIN_OBS_STD = 0.1  # a feature that is nearly always 0 must not be blown up
MAX_NORMALIZED_OBS = 5.0


def layer(n_in: int, n_out: int, gain: float = 2**0.5) -> nn.Linear:
    """A linear layer with orthogonal weights and zero bias, as PPO is usually set up."""
    linear = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(linear.weight, gain)
    nn.init.zeros_(linear.bias)
    return linear


class ActorCritic(nn.Module):
    """Two MLPs on the standardized state: the policy's logits and the state's value."""

    def __init__(self, n_obs: int, n_actions: int, model_config: ModelConfig) -> None:
        super().__init__()
        h1, h2 = model_config.hidden_sizes
        self.register_buffer("obs_mean", torch.zeros(n_obs))
        self.register_buffer("obs_var", torch.ones(n_obs))
        self.register_buffer("obs_count", torch.tensor(1e-4))
        self.actor = nn.Sequential(
            layer(n_obs, h1),
            nn.Tanh(),
            layer(h1, h2),
            nn.Tanh(),
            layer(h2, n_actions, gain=0.01),  # near-uniform policy at the start
        )
        self.critic = nn.Sequential(
            layer(n_obs, h1),
            nn.Tanh(),
            layer(h1, h2),
            nn.Tanh(),
            layer(h2, 1, gain=1.0),
        )

    @torch.no_grad()
    def update_normalization(self, states: torch.Tensor) -> None:
        """Fold a batch of states (B, n_obs) into the running mean and variance."""
        batch_mean, batch_var, batch_count = (
            states.mean(0),
            states.var(0),
            states.shape[0],
        )
        total = self.obs_count + batch_count
        delta = batch_mean - self.obs_mean
        self.obs_var.copy_(
            (
                self.obs_var * self.obs_count
                + batch_var * batch_count
                + delta**2 * self.obs_count * batch_count / total
            )
            / total
        )
        self.obs_mean.add_(delta * batch_count / total)
        self.obs_count.copy_(total)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        std = self.obs_var.sqrt().clamp(min=MIN_OBS_STD)
        return ((x - self.obs_mean) / std).clamp(-MAX_NORMALIZED_OBS, MAX_NORMALIZED_OBS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalize(x))

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self.normalize(x)).squeeze(-1)

    def masked_logits_and_value(
        self, states: torch.Tensor, masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits with the illegal actions masked out (B, n_actions), and values (B,)."""
        x = self.normalize(states)
        logits = self.actor(x).masked_fill(~masks, MASKED_LOGIT)
        return logits, self.critic(x).squeeze(-1)

    def action_and_value(
        self,
        states: torch.Tensor,
        masks: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draw legal actions (or take ``actions``): actions, log-probs, entropies, values.

        Shapes:
            states: (B, n_obs)   masks: (B, n_actions) bool   actions: (B,)
            returns: four tensors of shape (B,)
        """
        logits, value = self.masked_logits_and_value(states, masks)
        log_probs = torch.log_softmax(logits, dim=-1)  # (B, n_actions)
        probs = log_probs.exp()
        if actions is None:
            actions = torch.multinomial(probs, 1).squeeze(-1)
        log_prob = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        entropy = -(probs * log_probs.masked_fill(~masks, 0.0)).sum(dim=-1)
        return actions, log_prob, entropy, value


def num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_spec(model: ActorCritic) -> dict[str, Any]:
    """Everything needed to rebuild ``model`` in PyTorch and load ``trained_ppo.pth``
    (a ``state_dict``) into it; written to ``train_config.json`` under ``"model"``."""
    n_obs = int(model.obs_mean.shape[0])
    n_actions = int(model.actor[-1].out_features)
    h1, h2 = model.actor[0].out_features, model.actor[2].out_features
    return {
        "class": "ActorCritic",
        "source": "src/proximal-policy-optimization/seddler/model.py",
        "weights": "trained_ppo.pth",
        "weights_format": "state_dict (torch.save(model.state_dict()))",
        "constructor": {
            "n_obs": n_obs,
            "n_actions": n_actions,
            "hidden_sizes": [h1, h2],
        },
        "actor": [
            f"Linear({n_obs}, {h1})",
            "Tanh",
            f"Linear({h1}, {h2})",
            "Tanh",
            f"Linear({h2}, {n_actions})",
        ],
        "critic": [
            f"Linear({n_obs}, {h1})",
            "Tanh",
            f"Linear({h1}, {h2})",
            "Tanh",
            f"Linear({h2}, 1)",
        ],
        "obs_normalization": {
            "formula": "clamp((x - obs_mean) / clamp(sqrt(obs_var), min=min_obs_std), "
            "-max_normalized_obs, max_normalized_obs)",
            "buffers": ["obs_mean", "obs_var", "obs_count"],  # in the state_dict
            "min_obs_std": MIN_OBS_STD,
            "max_normalized_obs": MAX_NORMALIZED_OBS,
        },
        "outputs": {
            "forward": f"logits (B, {n_actions}) of the normalized input",
            "value": "state value (B,)",
        },
        "action_masking": {"illegal_logit": MASKED_LOGIT},
        "state_dict_shapes": {k: list(v.shape) for k, v in model.state_dict().items()},
        "num_parameters": num_parameters(model),
        "torch_version": torch.__version__,
    }


def load_model(run_dir: str | Path, device: str | torch.device = "cpu") -> ActorCritic:
    """Rebuild a run's ``ActorCritic`` from its ``train_config.json`` and load its weights."""
    run_dir = Path(run_dir)
    spec = json.loads((run_dir / "train_config.json").read_text())["model"]
    args = spec["constructor"]
    model_config = ModelConfig(hidden_sizes=tuple(args["hidden_sizes"]))
    model = ActorCritic(args["n_obs"], args["n_actions"], model_config)
    model.load_state_dict(torch.load(run_dir / spec["weights"], map_location=device))
    return model.to(device).eval()
