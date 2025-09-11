"""Utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector.utils import batch_space, create_empty_array
from gymnasium.vector.vector_env import VectorEnv


class MultiEnvWrapper(VectorEnv):
    """A vectorized environment wrapper that parallelizes multiple Gym
    environments.

    This wrapper creates multiple copies of the same environment and batches their
    observations, actions, rewards, and done signals for efficient vectorized RL
    training. Optionally supports PyTorch tensor conversion for seamless integration
    with neural network training.
    Follows gymnasium's VectorEnv interface but implemented as a synchronous wrapper.

    Args:
        env_fn: A callable that creates a single environment instance
        num_envs: Number of parallel environments to create
        auto_reset: Whether to automatically reset terminated environments
            (default: True)
        to_tensor: If True, observations and returns will be converted to PyTorch
            tensors, and tensor actions will be accepted (default: False)
        device: Device to place tensors on if to_tensor=True (default: "cpu")

    Example:
        >>> import prbench
        >>> prbench.register_all_environments()
        >>> env_fn = lambda: prbench.make("prbench/StickButton2D-b5-v0")
        >>> multi_env = MultiEnvWrapper(env_fn, num_envs=4)
        >>> obs_batch, info_batch = multi_env.reset(seed=123)
        >>> obs_batch.shape
        (4, observation_dim)
        >>> actions = multi_env.action_space.sample()
        >>> obs_batch, rewards, terminated, truncated, info_batch = multi_env.step(
        ...     actions
        ... )

        With tensor support:
        >>> multi_env = MultiEnvWrapper(env_fn, num_envs=4,
            to_tensor=True, device="cuda")
        >>> obs_batch, _ = multi_env.reset()  # Returns torch.Tensor on cuda
        >>> actions = torch.randn((4, action_dim), device="cuda")
        >>> obs, rewards, done, truncated, _ = multi_env.step(actions)  # All tensors
    """

    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        num_envs: int,
        auto_reset: bool = True,
        to_tensor: bool = False,
        device: str = "cpu",
    ):
        self.env_fn = env_fn
        self.auto_reset = auto_reset
        self.to_tensor = to_tensor
        self.device = device

        # Create all sub-environments
        self.envs = [env_fn() for _ in range(num_envs)]

        # Initialize VectorEnv attributes
        self.num_envs = num_envs
        self.observation_space = batch_space(self.envs[0].observation_space, num_envs)
        self.action_space = batch_space(self.envs[0].action_space, num_envs)

        # Store single environment spaces for reference
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space

        # Validate that all environments have the same spaces
        for i, env in enumerate(self.envs):
            assert env.action_space == self.single_action_space, (
                f"Environment {i} has different action space: {env.action_space} "
                f"vs expected {self.single_action_space}"
            )
            assert env.observation_space == self.single_observation_space, (
                f"Environment {i} has different observation space: "
                f"{env.observation_space} vs expected {self.single_observation_space}"
            )

        # Set metadata
        self.metadata = self.envs[0].metadata.copy()
        self.metadata["autoreset_mode"] = "next_step" if auto_reset else "disabled"

        # Initialize storage for batched data - use numpy array directly
        if isinstance(self.single_observation_space, gym.spaces.Box):
            self._observations = np.zeros(
                (num_envs,) + self.single_observation_space.shape,
                dtype=self.single_observation_space.dtype,
            )
        else:
            self._observations = create_empty_array(
                self.single_observation_space, n=num_envs, fn=np.zeros
            )  # type: ignore
        self._rewards = np.zeros((num_envs,), dtype=np.float64)
        self._terminations = np.zeros((num_envs,), dtype=np.bool_)
        self._truncations = np.zeros((num_envs,), dtype=np.bool_)
        self._env_needs_reset = np.ones((num_envs,), dtype=np.bool_)

    def _to_tensor(self, array: np.ndarray) -> Union[np.ndarray, torch.Tensor]:
        """Convert numpy array to tensor if to_tensor is True."""
        if self.to_tensor:
            return torch.from_numpy(array).to(self.device)
        return array

    def _to_numpy(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert tensor to numpy array if needed."""
        if torch.is_tensor(data):
            return data.detach().cpu().numpy()
        return data

    def reset(
        self, *, seed: int | Sequence[int] | None = None, options: dict | None = None
    ) -> tuple[Union[np.ndarray, torch.Tensor], dict]:
        """Reset all environments and return batched observations.

        Args:
            seed: Random seed(s) for the environments
            options: Environment-specific options

        Returns:
            Batched observations (as tensor if to_tensor=True) and info dict
        """
        # Handle seeding
        if seed is not None:
            if isinstance(seed, int):
                # Generate different seeds for each environment
                seeds_final: list[int | None] = [seed + i for i in range(self.num_envs)]
            else:
                assert (
                    len(seed) == self.num_envs
                ), f"Seed list length {len(seed)} doesn't match num_envs {self.num_envs}"
                seeds_final = list(seed)
        else:
            seeds_final = [None] * self.num_envs

        # Reset all environments
        infos: dict = {}
        for i, (env, env_seed) in enumerate(zip(self.envs, seeds_final)):
            obs, info = env.reset(seed=env_seed, options=options)
            self._observations[i] = obs

            # Accumulate info dicts - create arrays for each key
            for key, value in info.items():
                if key not in infos:
                    infos[key] = [None] * self.num_envs  # type: ignore
                infos[key][i] = value

        # Convert info lists to arrays where possible
        for key, value_list in infos.items():
            try:
                infos[key] = np.array(value_list)  # type: ignore
            except (ValueError, TypeError):
                # Keep as list if can't convert to array
                pass

        # Reset tracking flags
        self._env_needs_reset.fill(False)
        self._terminations.fill(False)
        self._truncations.fill(False)

        observations = np.array(self._observations)
        return self._to_tensor(observations), infos

    def step(self, actions: Union[np.ndarray, torch.Tensor]) -> tuple[
        Union[np.ndarray, torch.Tensor],
        Union[np.ndarray, torch.Tensor],
        Union[np.ndarray, torch.Tensor],
        Union[np.ndarray, torch.Tensor],
        dict,
    ]:
        """Step all environments with batched actions.

        Args:
            actions: Batched actions with shape (num_envs, action_dim)
                    Can be numpy array or torch tensor

        Returns:
            Batched observations, rewards, terminations, truncations, and infos
            Returns tensors if to_tensor=True, otherwise numpy arrays
        """
        # Convert actions to numpy if they are tensors
        actions_np = self._to_numpy(actions)
        assert self.action_space.contains(actions_np), "Actions not in action space"

        infos: dict = {}

        for i, env in enumerate(self.envs):
            # Handle auto-reset after termination/truncation
            if self._env_needs_reset[i] and self.auto_reset:
                obs, reset_info = env.reset()
                self._observations[i] = obs
                # Use reset observation and zero reward for auto-reset
                self._rewards[i] = 0.0
                self._terminations[i] = False
                self._truncations[i] = False
                self._env_needs_reset[i] = False

                # Add reset info to the batch
                for key, value in reset_info.items():
                    if key not in infos:
                        infos[key] = [None] * self.num_envs  # type: ignore
                    infos[key][i] = value
                continue

            # Step the environment normally
            action = actions_np[i]
            obs, reward, terminated, truncated, info = env.step(action)

            # Store results
            self._observations[i] = obs
            self._rewards[i] = reward
            self._terminations[i] = terminated
            self._truncations[i] = truncated

            # Mark environment for reset if done
            if terminated or truncated:
                self._env_needs_reset[i] = True

            # Accumulate info
            for key, value in info.items():
                if key not in infos:
                    infos[key] = [None] * self.num_envs  # type: ignore
                infos[key][i] = value

        # Convert info lists to arrays where possible
        for key, value_list in infos.items():
            try:
                infos[key] = np.array(value_list)  # type: ignore
            except (ValueError, TypeError):
                # Keep as list if can't convert to array
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

    def seed(self, seeds: int | Sequence[int] | None = None):
        """Seed all environments.

        Args:
            seeds: Single seed or sequence of seeds for each environment
        """
        if seeds is None:
            seeds_list: list[int | None] = [None] * self.num_envs
        elif isinstance(seeds, int):
            seeds_list = [seeds + i for i in range(self.num_envs)]
        else:
            assert len(seeds) == self.num_envs, (
                f"Seed sequence length {len(seeds)} doesn't match num_envs "
                f"{self.num_envs}"
            )
            seeds_list = list(seeds)  # type: ignore

        for env, seed in zip(self.envs, seeds_list):
            if hasattr(env.action_space, "seed"):
                env.action_space.seed(seed)

    @property
    def unwrapped(self):
        """Return the underlying environments."""
        return self.envs
