import os
import torch
import gymnasium as gym

from policies.base import BasePolicy

try:
    from stable_baselines3 import PPO, SAC, TD3
    _SB3_AVAILABLE = True
except ImportError:
    print("MOCK MODE")
    _SB3_AVAILABLE = False

try:
    from huggingface_hub import hf_hub_download
    _HF_HUB_AVAILABLE = True
except ImportError:
    _HF_HUB_AVAILABLE = False


class SB3HubPolicyWrapper(BasePolicy):
    """
    Loads Stable-Baselines3 policies from Hugging Face Hub repositories or
    local directories, and ensures they are structurally compatible with the
    active Isaac Lab environment's observation and action spaces.
    """
    def __init__(self, repo_id: str, filename: str, env: gym.Env):
        self.env = env
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.checkpoint_path = self._resolve_checkpoint(repo_id, filename)
        self.model = self._load_policy_checkpoint()

        super().__init__(self.model.observation_space, self.model.action_space)

        self._assert_space_compatibility()

    def _resolve_checkpoint(self, repo_id: str, filename: str) -> str:
        if os.path.exists(os.path.join(repo_id, filename)):
            print(f"[Loader] Found local checkpoint path layout for: {filename}")
            return os.path.join(repo_id, filename)
        elif os.path.exists(filename):
            return filename

        if _HF_HUB_AVAILABLE:
            try:
                print(f"[Loader] Syncing with Hugging Face Hub repo: '{repo_id}'...")
                return hf_hub_download(repo_id=repo_id, filename=filename)
            except Exception as e:
                raise IOError(f"Failed pulling asset '{filename}' from HF Hub repository '{repo_id}'. Detail: {e}")
        else:
            raise ImportError(
                "Requested remote HF Hub path checkpoint target, but `huggingface_hub` package "
                "is completely missing from this Python runtime execution context."
            )

    def _load_policy_checkpoint(self):
        if not _SB3_AVAILABLE:
            class MockModel:
                observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
                action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
            return MockModel()

        lower_name = self.checkpoint_path.lower()
        if "sac" in lower_name:
            return SAC.load(self.checkpoint_path, device=self.device)
        elif "td3" in lower_name:
            return TD3.load(self.checkpoint_path, device=self.device)
        else:
            return PPO.load(self.checkpoint_path, device=self.device)

    def _get_space_shape(self, space) -> int:
        if hasattr(space, "shape") and space.shape is not None:
            return space.shape[-1] if len(space.shape) > 0 else 1
        elif hasattr(space, "flat_dim"):
            return space.flat_dim
        raise AttributeError(f"Could not automatically resolve spatial dimensions of object type: {type(space)}")

    def _assert_space_compatibility(self):
        if not _SB3_AVAILABLE:
            return

        try:
            env_obs_shape = self._get_space_shape(self.env.observation_space)
            model_obs_shape = self._get_space_shape(self.model.observation_space)
            env_act_shape = self._get_space_shape(self.env.action_space)
            model_act_shape = self._get_space_shape(self.model.action_space)
        except Exception as e:
            raise RuntimeError(f"CRITICAL INTERCEPT: Details: {e}")

        if type(self.env.action_space) is not type(self.model.action_space):
            raise ValueError(
                f"\n{'='*80}\n"
                f"CRITICAL TYPE MISMATCH AGAINST TASK ACTION SPACE\n"
                f"{'='*80}\n"
                f" Active Environment Action Space Type: {type(self.env.action_space).__name__}\n"
                f" Loaded Checkpoint Action Space Type: {type(self.model.action_space).__name__}\n"
                f" Cause: Shapes may coincidentally match while the underlying action type differs\n"
                f" (e.g. Discrete vs Box). Dimension-only checks cannot catch this.\n"
                f"{'='*80}"
            )

        if env_obs_shape != model_obs_shape:
            raise ValueError(
                f"\n{'='*80}\n"
                f"CRITICAL MISMATCH DETECTED AGAINST TASK OBSERVATION SPACE\n"
                f"{'='*80}\n"
                f" Active Environment Observation Dimension: {env_obs_shape}\n"
                f" Loaded Checkpoint Policy Architecture: {model_obs_shape}\n"
                f" Cause: You are feeding an invalid policy vector size into your target task pipeline.\n"
                f"{'='*80}"
            )

        if env_act_shape != model_act_shape:
            raise ValueError(
                f"\n{'='*80}\n"
                f"CRITICAL MISMATCH DETECTED AGAINST TASK ACTION SPACE\n"
                f"{'='*80}\n"
                f" Active Environment Motor DoF: {env_act_shape}\n"
                f" Loaded Checkpoint Output Layers: {model_act_shape}\n"
                f" Cause: Motor controls do not align with loaded weight dimension arrays.\n"
                f"{'='*80}"
            )

    def predict(self, observations: torch.Tensor, deterministic: bool = True) -> tuple:
        """Predicts actions based on the provided observations using the loaded policy."""
        
        if not _SB3_AVAILABLE:
            batch_size = observations.shape[0] if len(observations.shape) > 1 else 1
            mock_action = torch.zeros((batch_size, self._get_space_shape(self.env.action_space)), device=observations.device)
            return mock_action, None

        if isinstance(observations, torch.Tensor):
            obs_np = observations.detach().cpu().numpy()
        else:
            obs_np = observations

        action_np, states = self.model.predict(obs_np, deterministic=deterministic)
        action = torch.as_tensor(action_np, device=self.device, dtype=torch.float32)
        return action, states


class DummyFrankaReachEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,))


if __name__ == "__main__":
    env = DummyFrankaReachEnv()
    print("Testing loader against an incompatible public repository on the Hub...")
    try:
        wrapper = SB3HubPolicyWrapper(
            repo_id="sb3/ppo-CartPole-v1",
            filename="ppo-CartPole-v1.zip",
            env=env
        )
    except ValueError as err:
        print("\nSUCCESS: Loader caught the problem and failed loudly as expected:")
        print(err)
