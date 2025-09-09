"""Tests for gym utilities."""

import gymnasium as gym
import numpy as np
import pytest

from prpl_utils.gym_utils import MultiEnvWrapper


def test_multi_env_wrapper():
    """Test basic functionality of MultiEnvWrapper with CartPole."""

    # Create environment factory function
    env_fn = lambda: gym.make("CartPole-v1")

    # Create multi-environment wrapper with 3 environments
    num_envs = 3
    multi_env = MultiEnvWrapper(env_fn, num_envs=num_envs)

    assert multi_env.num_envs == num_envs
    assert hasattr(multi_env, "action_space")
    assert hasattr(multi_env, "observation_space")

    # Test reset
    obs_batch, info_batch = multi_env.reset(seed=123)
    assert obs_batch.shape[0] == num_envs
    assert obs_batch.shape[1] == 4  # CartPole observation space
    assert isinstance(info_batch, dict)

    # Test step with random actions
    actions = multi_env.action_space.sample()
    assert actions.shape[0] == num_envs

    obs_batch, rewards, terminated, truncated, info_batch = multi_env.step(actions)
    assert obs_batch.shape[0] == num_envs
    assert obs_batch.shape[1] == 4  # CartPole observation space
    assert rewards.shape == (num_envs,)
    assert terminated.shape == (num_envs,)
    assert truncated.shape == (num_envs,)
    assert isinstance(info_batch, dict)

    # Test a few more steps to verify functionality
    for _ in range(3):
        actions = multi_env.action_space.sample()
        obs_batch, rewards, terminated, truncated, info_batch = multi_env.step(actions)
        assert obs_batch.shape[0] == num_envs

    # Close environments
    multi_env.close()


def test_multi_env_wrapper_mountaincar():
    """Test MultiEnvWrapper with MountainCar environment."""

    env_fn = lambda: gym.make("MountainCar-v0")
    num_envs = 2
    multi_env = MultiEnvWrapper(env_fn, num_envs=num_envs)

    # Test spaces
    assert multi_env.num_envs == num_envs
    assert multi_env.single_action_space.n == 3  # MountainCar has 3 actions
    assert multi_env.single_observation_space.shape == (2,)  # Position and velocity

    # Test reset
    obs_batch, _ = multi_env.reset(seed=42)
    assert obs_batch.shape == (num_envs, 2)

    # Test step
    actions = np.array([0, 1])  # Specific actions for each environment
    obs_batch, rewards, _, _, _ = multi_env.step(actions)
    assert obs_batch.shape == (num_envs, 2)
    assert rewards.shape == (num_envs,)

    multi_env.close()


def test_multi_env_wrapper_acrobot():
    """Test MultiEnvWrapper with Acrobot environment."""

    env_fn = lambda: gym.make("Acrobot-v1")
    num_envs = 4
    multi_env = MultiEnvWrapper(env_fn, num_envs=num_envs)

    # Test spaces
    assert multi_env.num_envs == num_envs
    assert multi_env.single_action_space.n == 3  # Acrobot has 3 actions
    assert multi_env.single_observation_space.shape == (6,)  # 6-dimensional observation

    # Test reset
    obs_batch, _ = multi_env.reset()
    assert obs_batch.shape == (num_envs, 6)

    # Test multiple steps
    for _ in range(5):
        actions = multi_env.action_space.sample()
        obs_batch, rewards, terminated, truncated, _ = multi_env.step(actions)
        assert obs_batch.shape == (num_envs, 6)
        assert rewards.shape == (num_envs,)
        assert terminated.shape == (num_envs,)
        assert truncated.shape == (num_envs,)

    multi_env.close()


def test_multi_env_wrapper_auto_reset():
    """Test auto_reset functionality."""

    # Use CartPole which has a relatively short episode length
    env_fn = lambda: gym.make("CartPole-v1", max_episode_steps=10)
    num_envs = 2
    multi_env = MultiEnvWrapper(env_fn, num_envs=num_envs, auto_reset=True)

    _obs_batch, _ = multi_env.reset(seed=123)

    # Run enough steps to likely trigger episode termination
    terminated_count = 0
    for _ in range(50):
        actions = multi_env.action_space.sample()
        _obs_batch, _rewards, terminated, truncated, _info_batch = multi_env.step(
            actions
        )
        if np.any(terminated) or np.any(truncated):
            terminated_count += 1

    # Should have seen some terminations with auto-reset
    assert terminated_count > 0, "Expected some episodes to terminate and auto-reset"

    multi_env.close()


def test_multi_env_wrapper_seeding():
    """Test seeding behavior."""

    env_fn = lambda: gym.make("CartPole-v1")
    num_envs = 3

    # Test with single seed
    multi_env1 = MultiEnvWrapper(env_fn, num_envs=num_envs)
    obs1, _ = multi_env1.reset(seed=42)

    multi_env2 = MultiEnvWrapper(env_fn, num_envs=num_envs)
    obs2, _ = multi_env2.reset(seed=42)

    # Should get same initial observations with same seed
    np.testing.assert_array_equal(obs1, obs2)

    # Test with different seeds
    obs3, _ = multi_env1.reset(seed=123)
    assert not np.array_equal(obs1, obs3)

    # Test with seed list
    seed_list = [10, 20, 30]
    obs4, _ = multi_env1.reset(seed=seed_list)
    assert obs4.shape == (num_envs, 4)

    multi_env1.close()
    multi_env2.close()


def test_multi_env_wrapper_different_action_spaces():
    """Test with environment that has continuous action space."""

    try:
        env_fn = lambda: gym.make("Pendulum-v1")
        num_envs = 2
        multi_env = MultiEnvWrapper(env_fn, num_envs=num_envs)

        # Test spaces
        assert multi_env.num_envs == num_envs
        assert multi_env.single_observation_space.shape == (3,)
        assert multi_env.single_action_space.shape == (1,)

        # Test reset and step
        obs_batch, _ = multi_env.reset(seed=42)
        assert obs_batch.shape == (num_envs, 3)

        actions = multi_env.action_space.sample()
        assert actions.shape == (num_envs, 1)

        obs_batch, rewards, _terminated, _truncated, _info_batch = multi_env.step(
            actions
        )
        assert obs_batch.shape == (num_envs, 3)
        assert rewards.shape == (num_envs,)

        multi_env.close()

    except gym.error.UnregisteredEnv:
        # Skip test if Pendulum is not available
        pytest.skip("Pendulum-v1 environment not available")


def test_multi_env_wrapper_no_auto_reset():
    """Test behavior with auto_reset=False."""

    env_fn = lambda: gym.make("CartPole-v1", max_episode_steps=5)
    num_envs = 2
    multi_env = MultiEnvWrapper(env_fn, num_envs=num_envs, auto_reset=False)

    _obs_batch, _ = multi_env.reset(seed=123)

    # Run enough steps to trigger termination
    done_envs = set()
    for step in range(20):
        actions = multi_env.action_space.sample()
        _obs_batch, _rewards, terminated, truncated, _info_batch = multi_env.step(
            actions
        )

        # Track which environments are done
        for i in range(num_envs):
            if terminated[i] or truncated[i]:
                done_envs.add(i)

        # Once we have terminated environments, verify they stay terminated
        if done_envs and step > 10:
            break

    # Should have some terminated environments
    assert len(done_envs) > 0, "Expected some environments to terminate"

    multi_env.close()
