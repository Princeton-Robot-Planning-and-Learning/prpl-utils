"""Utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector.utils import batch_space, create_empty_array


class MultiEnvWrapper(gym.Env):
    """A batched single-Env wrapper over multiple Gym environments.

    This class exposes N identical sub-environments as ONE `gym.Env` whose
    observation_space and action_space are the batched versions of the
    single-env spaces (via `batch_space`). This makes it compatible with
    wrappers like `gymnasium.wrappers.RecordVideo` that expect `gym.Env`,
    while still enabling vectorized stepping.

    It supports optional PyTorch tensor IO for seamless RL training, and a
    tiled `rgb_array` render for video recording.

    Args:
        env_fn: A callable that creates a single environment instance
        num_envs: Number of sub-environments to create
        auto_reset: Whether to automatically reset terminated environments
            (default: True)
        to_tensor: If True, observations and returns will be converted to PyTorch
            tensors, and tensor actions will be accepted (default: False)
        device: Device to place tensors on if to_tensor=True (default: "cpu")
        render_mode: Render mode; should be "rgb_array" to use RecordVideo

    Example:
        >>> import prbench
        >>> prbench.register_all_environments()
        >>> env_fn = lambda: prbench.make("prbench/StickButton2D-b5-v0", render_mode="rgb_array")
        >>> multi_env = MultiEnvWrapper(env_fn, num_envs=4, render_mode="rgb_array")
        >>> obs_batch, info_batch = multi_env.reset(seed=123)
        >>> obs_batch.shape
        (4, observation_dim)
        >>> actions = multi_env.action_space.sample()
        >>> obs_batch, rewards, terminated, truncated, info_batch = multi_env.step(actions)

        With tensor support:
        >>> multi_env = MultiEnvWrapper(env_fn, num_envs=4, to_tensor=True, device="cuda", render_mode="rgb_array")
        >>> obs_batch, _ = multi_env.reset()  # returns torch.Tensor on cuda
        >>> actions = torch.randn((4, *multi_env.single_action_space.shape), device="cuda")
        >>> obs, rewards, done, truncated, _ = multi_env.step(actions)
    """

    # Make sure RecordVideo recognizes rgb_array rendering
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        num_envs: int,
        auto_reset: bool = True,
        to_tensor: bool = False,
        device: str = "cpu",
        render_mode: str | None = "rgb_array",
    ):
        super().__init__()
        self.env_fn = env_fn
        self.num_envs = int(num_envs)
        assert self.num_envs >= 1, "num_envs must be >= 1"
        self.auto_reset = auto_reset
        self.to_tensor = to_tensor
        self.device = device
        self.render_mode = render_mode

        # Create all sub-environments
        # TIP: Prefer env_fn that accepts render_mode="rgb_array" for recording.
        self.envs = [env_fn() for _ in range(self.num_envs)]

        # Spaces
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        # Validate homogeneous spaces
        for i, env in enumerate(self.envs):
            assert env.action_space == self.single_action_space, (
                f"Environment {i} has different action space: {env.action_space} "
                f"vs expected {self.single_action_space}"
            )
            assert env.observation_space == self.single_observation_space, (
                f"Environment {i} has different observation space: "
                f"{env.observation_space} vs expected {self.single_observation_space}"
            )

        # Buffers
        if isinstance(self.single_observation_space, gym.spaces.Box):
            self._observations = np.zeros(
                (self.num_envs,) + self.single_observation_space.shape,
                dtype=self.single_observation_space.dtype,
            )
        else:
            self._observations = create_empty_array(
                self.single_observation_space, n=self.num_envs, fn=np.zeros  # type: ignore
            )  # type: ignore
        self._rewards = np.zeros((self.num_envs,), dtype=np.float32)
        self._terminations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._env_needs_reset = np.ones((self.num_envs,), dtype=np.bool_)

        # Copy metadata and annotate autoreset status
        self.metadata = dict(getattr(self.envs[0], "metadata", {}))
        self.metadata["render_modes"] = list(
            set(self.metadata.get("render_modes", []) + ["rgb_array"])
        )
        self.metadata["autoreset_mode"] = "next_step" if auto_reset else "disabled"

    # ------------------------- utilities -------------------------

    def _to_tensor(self, array: np.ndarray) -> Union[np.ndarray, torch.Tensor]:
        if self.to_tensor:
            return torch.from_numpy(array).to(self.device)
        return array

    def _to_numpy(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if torch.is_tensor(data):
            return data.detach().cpu().numpy()
        return data

    # --------------------------- API -----------------------------

    def reset(
        self, *, seed: int | Sequence[int] | None = None, options: dict | None = None
    ) -> tuple[Union[np.ndarray, torch.Tensor], dict]:
        """Reset all sub-environments and return batched observation and
        info."""
        # Distribute seeds
        if seed is not None:
            if isinstance(seed, int):
                seeds_final: list[int | None] = [seed + i for i in range(self.num_envs)]
            else:
                seed = list(seed)
                assert (
                    len(seed) == self.num_envs
                ), f"Seed list length {len(seed)} doesn't match num_envs {self.num_envs}"
                seeds_final = seed  # type: ignore
        else:
            seeds_final = [None] * self.num_envs

        # Reset
        infos: dict[str, Any] = {}
        for i, (env, env_seed) in enumerate(zip(self.envs, seeds_final)):
            obs, info = env.reset(seed=env_seed, options=options)
            # Write obs into buffer
            if isinstance(self.single_observation_space, gym.spaces.Box):
                self._observations[i] = obs
            else:
                # For Dict/Tuple/etc. spaces, rely on create_empty_array structure
                self._observations[i] = obs  # type: ignore[assignment]

            # Batch info
            for key, value in info.items():
                if key not in infos:
                    infos[key] = [None] * self.num_envs  # type: ignore
                infos[key][i] = value

        # Convert info lists to arrays when possible
        for key, value_list in list(infos.items()):
            try:
                infos[key] = np.array(value_list)  # type: ignore
            except (ValueError, TypeError):
                pass

        # Reset trackers
        self._env_needs_reset.fill(False)
        self._terminations.fill(False)
        self._truncations.fill(False)
        self._rewards.fill(0.0)

        observations = np.array(self._observations)
        return self._to_tensor(observations), infos

    def step(self, actions: Union[np.ndarray, torch.Tensor]) -> tuple[
        Union[np.ndarray, torch.Tensor],
        Union[np.ndarray, torch.Tensor],
        Union[np.ndarray, torch.Tensor],
        Union[np.ndarray, torch.Tensor],
        dict,
    ]:
        """Step all sub-environments with batched actions."""
        actions_np = self._to_numpy(actions)
        assert self.action_space.contains(actions_np), "Actions not in action space"

        infos: dict[str, Any] = {}

        for i, env in enumerate(self.envs):
            # Auto-reset paths
            if self._env_needs_reset[i] and self.auto_reset:
                obs, reset_info = env.reset()
                if isinstance(self.single_observation_space, gym.spaces.Box):
                    self._observations[i] = obs
                else:
                    self._observations[i] = obs  # type: ignore[assignment]
                self._rewards[i] = 0.0
                self._terminations[i] = False
                self._truncations[i] = False
                self._env_needs_reset[i] = False

                for key, value in reset_info.items():
                    if key not in infos:
                        infos[key] = [None] * self.num_envs  # type: ignore
                    infos[key][i] = value
                continue

            # Normal step
            action = actions_np[i]
            obs, reward, terminated, truncated, info = env.step(action)

            if isinstance(self.single_observation_space, gym.spaces.Box):
                self._observations[i] = obs
            else:
                self._observations[i] = obs  # type: ignore[assignment]
            self._rewards[i] = np.float32(reward)
            self._terminations[i] = bool(terminated)
            self._truncations[i] = bool(truncated)

            # If done, mark for auto-reset next call; also stash finals per Gymnasium convention
            if terminated or truncated:
                info.setdefault("final_observation", obs)
                info.setdefault("final_info", info.copy())
                self._env_needs_reset[i] = True

            for key, value in info.items():
                if key not in infos:
                    infos[key] = [None] * self.num_envs  # type: ignore
                infos[key][i] = value

        # Convert info lists to arrays where possible
        for key, value_list in list(infos.items()):
            try:
                infos[key] = np.array(value_list)  # type: ignore
            except (ValueError, TypeError):
                pass

        observations = np.array(self._observations)
        rewards = self._rewards.copy()
        terminations = self._terminations.copy()
        truncations = self._truncations.copy()

        return (
            self._to_tensor(observations),
            self._to_tensor(rewards),
            self._to_tensor(terminations),
            self._to_tensor(truncations),
            infos,
        )

    def render(self) -> np.ndarray | None:  # type: ignore
        """Render all environments and tile them in a 4x4 grid.

        Returns:
            Tiled image as numpy array with shape (height, width, 3) or None
        """
        results: list[np.ndarray] = []
        for env in self.envs:
            result: np.ndarray | None = env.render()  # type: ignore
            if result is not None:
                results.append(result)  # type: ignore

        if not results:
            return None

        # Tile images in a 4x4 grid (max 16 environments)
        max_envs = min(len(results), 16)
        results = results[:max_envs]

        # Calculate grid dimensions
        grid_cols = min(4, max_envs)
        grid_rows = (max_envs + grid_cols - 1) // grid_cols

        # Get dimensions from first image
        img_height, img_width = results[0].shape[:2]
        channels = results[0].shape[2] if len(results[0].shape) == 3 else 1

        # Create tiled image
        tiled_height = grid_rows * img_height
        tiled_width = grid_cols * img_width

        if channels == 1:
            tiled_image = np.zeros((tiled_height, tiled_width), dtype=results[0].dtype)
        else:
            tiled_image = np.zeros(
                (tiled_height, tiled_width, channels), dtype=results[0].dtype
            )

        # Fill tiled image
        for i, img in enumerate(results):
            row = i // grid_cols
            col = i % grid_cols

            start_row = row * img_height
            end_row = start_row + img_height
            start_col = col * img_width
            end_col = start_col + img_width

            if channels == 1:
                tiled_image[start_row:end_row, start_col:end_col] = img
            else:
                tiled_image[start_row:end_row, start_col:end_col] = img

        return tiled_image

    def close(self, **kwargs):
        """Close all environments."""
        del kwargs  # Unused parameter required by VectorEnv interface
        for env in self.envs:
            if hasattr(env, "close"):
                env.close()

    # ----------------------- seeding helpers ---------------------

    def seed(self, seeds: int | Sequence[int] | None = None):
        """Seed sub-envs.

        Prefer `reset(seed=...)` for Gymnasium compatibility.
        """
        if seeds is None:
            seeds_list: list[int | None] = [None] * self.num_envs
        elif isinstance(seeds, int):
            seeds_list = [seeds + i for i in range(self.num_envs)]
        else:
            seeds = list(seeds)
            assert (
                len(seeds) == self.num_envs
            ), f"Seed sequence length {len(seeds)} doesn't match num_envs {self.num_envs}"
            seeds_list = seeds  # type: ignore

        for env, s in zip(self.envs, seeds_list):
            # Best-effort: Gymnasium prefers `reset(seed=...)`; some spaces still allow action_space.seed
            try:
                env.reset(seed=s)
            except Exception:
                pass
            if hasattr(env.action_space, "seed"):
                env.action_space.seed(s)

    # -------------------------- misc -----------------------------

    @property
    def unwrapped(self):
        """Return the underlying sub-environments list."""
        return self.envs
