class PPOSpec:
    def __init__(self, env_id, timesteps, device, search_space, reward_wrapper, reward_kwargs):
        self.env_id = env_id
        self.timesteps = timesteps
        self.device = device
        self.search_space = search_space
        self.reward_wrapper = reward_wrapper
        self.reward_kwargs = reward_kwargs

    @classmethod
    def from_sweep_definition(cls, sweep):
        return cls(
            env_id=sweep["env_id"],
            timesteps=sweep["timesteps"],
            device=sweep.get("device", "cpu"),
            search_space=sweep["search_space"],
            reward_wrapper=sweep["reward_wrapper"],
            reward_kwargs=sweep["reward_kwargs"],
        )

    def sample_hyperparams(self, trial): 
        hp = {}
        for k, spec in self.search_space.items():
            kind, *vals = spec
            if kind == "loguniform":
                hp[k] = trial.suggest_float(k, vals[0], vals[1], log=True)
            elif kind == "uniform":
                hp[k] = trial.suggest_float(k, vals[0], vals[1])
            elif kind == "categorical":
                hp[k] = trial.suggest_categorical(k, vals[0])
        return hp

    def build_exp(self, hparams):
        return {
            "backend": "sb3",
            "algorithm": "ppo",
            "env_id": self.env_id,
            "timesteps": self.timesteps,
            "device": self.device,
            "reward_wrapper": self.reward_wrapper,
            "reward_kwargs": self.reward_kwargs,
            **hparams,
        }

    def train(self, exp, run_dir):
        from src.trainers.sb3.ppo import train
        train(exp, run_dir)
