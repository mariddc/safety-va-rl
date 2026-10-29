SWEEPS = {
    "sac_goal2_sweep": {
        "backend": "sb3",
        "algorithm": "sac",
        "env_id": "SafetyPointGoal2-v0",
        "timesteps": 500_000,
        "device": "cuda",
        "reward_wrapper": "native",
        "reward_kwargs": {},

        "objective": "reward_minus_cost",
        "lambda_cost": 1.0,

        "n_trials": 8,

        "search_space": {
            "learning_rate": ("loguniform", 1e-4, 3e-3),
            "batch_size": ("categorical", [256, 512, 1024]),
            "gamma": ("uniform", 0.95, 0.999),
            "tau": ("uniform", 0.005, 0.02),
        },
    },

    "sac_goal1_sweep_100k": {
        "backend": "sb3",
        "algorithm": "sac",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 100_000,
        "device": "cuda",
        "reward_wrapper": "native",
        "reward_kwargs": {},

        "objective": "reward_minus_cost",
        "lambda_cost": 1.0,

        "n_trials": 8,

        "search_space": {
            "learning_rate": ("loguniform", 1e-4, 3e-3),
            "batch_size": ("categorical", [256, 512, 1024]),
            "gamma": ("uniform", 0.95, 0.999),
            "tau": ("uniform", 0.005, 0.02),
        },
    },

    "sac_goal1_sweep_10k": {
        "backend": "sb3",
        "algorithm": "sac",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 10_000,
        "device": "cuda",
        "reward_wrapper": "native",
        "reward_kwargs": {},

        "objective": "reward_minus_cost",
        "lambda_cost": 1.0,

        "n_trials": 8,

        "search_space": {
            "learning_rate": ("loguniform", 1e-4, 3e-3),
            "batch_size": ("categorical", [256, 512, 1024]),
            "gamma": ("uniform", 0.95, 0.999),
            "tau": ("uniform", 0.005, 0.02),
        },
    },

    "sac_goal2_sweep_100k": {
        "backend": "sb3",
        "algorithm": "sac",
        "env_id": "SafetyPointGoal2-v0",
        "timesteps": 100_000,
        "device": "cuda",
        "reward_wrapper": "native",
        "reward_kwargs": {},

        "objective": "reward_minus_cost",
        "lambda_cost": 1.0,

        "n_trials": 8,

        "search_space": {
            "learning_rate": ("loguniform", 1e-4, 3e-3),
            "batch_size": ("categorical", [256, 512, 1024]),
            "gamma": ("uniform", 0.95, 0.999),
            "tau": ("uniform", 0.005, 0.02),
        },
    },

    "sac_goal1_sweep_1m": {
        "backend": "sb3",
        "algorithm": "sac",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 1_000_000,
        "device": "cuda",
        "reward_wrapper": "native",
        "reward_kwargs": {},

        "objective": "reward_minus_cost",
        "lambda_cost": 1.0,

        "n_trials": 5,

        "search_space": {
            "learning_rate": ("loguniform", 1e-4, 3e-3),
            "batch_size": ("categorical", [256, 512, 1024]),
            "gamma": ("uniform", 0.95, 0.999),
            "tau": ("uniform", 0.005, 0.02),
        },

    },

    "ppo_goal1_sweep_10k": {
        "backend": "sb3",
        "algorithm": "ppo",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 10_000,
        "device": "cuda",
        "reward_wrapper": "native",
        "reward_kwargs": {},

        "objective": "reward_minus_cost",
        "lambda_cost": 1.0,

        "n_trials": 8,

        "search_space": {
            "learning_rate": ("loguniform", 1e-5, 3e-4),
            "n_steps": ("categorical", [2048]),
            "batch_size": ("categorical", [128, 256]),
            "gamma": ("uniform", 0.98, 0.999),
            "clip_range": ("uniform", 0.15, 0.25),
        }
    },
}
