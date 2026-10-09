EXPERIMENTS = {
    "sac_10k": {
        "backend": "sb3",
        "algorithm": "sac",
        "env_id": "SafetyPointGoal2-v0",
        "timesteps": 10_000,
        "seed": 0,
        "n_envs": 8,
    },

    "sac_400k": {
        "backend": "sb3",
        "algorithm": "sac",
        "env_id": "SafetyPointGoal2-v0",
        "timesteps": 400_000,
        "seed": 0,
        "n_envs": 8
    },

    "ppo_400k": {
        "backend": "sb3",
        "algorithm": "ppo",
        "env_id": "SafetyPointGoal2-v0",
        "timesteps": 400_000,
        "seed": 0,
        "n_envs": 8,
    },

    "sac_700k_goal1": {
        "backend": "sb3",
        "algorithm": "sac",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 700_000,
        "seed": 0,
        "n_envs": 8
    },

    "ppo_700k_goal1": {
        "backend": "sb3",
        "algorithm": "ppo",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 700_000,
        "seed": 0,
        "n_envs": 8,
    },

    "sac_1m500k_goal1": {
        "backend": "sb3",
        "algorithm": "sac",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 1_500_000,
        "seed": 0,
        "n_envs": 8
    },

    "ppo_1m500k_goal1": {
        "backend": "sb3",
        "algorithm": "ppo",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 1_500_000,
        "seed": 0,
        "n_envs": 8,
    },

    "ppo_1m200k_goal1": {
        "backend": "sb3",
        "algorithm": "ppo",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 1_200_000,
        "seed": 0,
        "n_envs": 8,
    },

    "ppo_3m600k_goal1_safe": {
        "backend": "sb3",
        "algorithm": "ppo",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 3_600_000,
        "seed": 0,
        "n_envs": 8,
    },

    "ppo_3m600k_goal2_safe": {
        "backend": "sb3",
        "algorithm": "ppo",
        "env_id": "SafetyPointGoal2-v0",
        "timesteps": 3_600_000,
        "seed": 0,
        "n_envs": 8,
    },

    "sac_3m600k_goal1_safe": {
        "backend": "sb3",
        "algorithm": "sac",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 3_600_000,
        "seed": 0,
        "n_envs": 8
    },

    "sac_3m600k_goal2_safe": {
        "backend": "sb3",
        "algorithm": "sac",
        "env_id": "SafetyPointGoal2-v0",
        "timesteps": 3_600_000,
        "seed": 0,
        "n_envs": 8
    },

    "sac_1m800k_goal1_safe": {
        "backend": "sb3",
        "algorithm": "sac",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 1_800_000,
        "seed": 0,
        "n_envs": 8
    },

    "sac_180k_goal1_safe": {
        "backend": "sb3",
        "algorithm": "sac",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 180_000,
        "seed": 0,
        "n_envs": 8
    },

    "sac_1m500k": {
        "backend": "sb3",
        "algorithm": "sac",
        "env_id": "SafetyPointGoal2-v0",
        "timesteps": 1_500_000,
        "seed": 0,
        "n_envs": 8,
    },

    "sac_2m500k": {
        "backend": "sb3",
        "algorithm": "sac",
        "env_id": "SafetyPointGoal2-v0",
        "timesteps": 2_500_000,
        "seed": 0,
        "n_envs": 8,
    },

    "sac_3m500k": {
        "backend": "sb3",
        "algorithm": "sac",
        "env_id": "SafetyPointGoal2-v0",
        "timesteps": 3_500_000,
        "seed": 0,
        "n_envs": 8,
    },

    "ppo_1m500k": {
        "backend": "sb3",
        "algorithm": "ppo",
        "env_id": "SafetyPointGoal2-v0",
        "timesteps": 1_500_000,
        "seed": 0,
        "n_envs": 8,
    },

    "ppo_2m500k": {
        "backend": "sb3",
        "algorithm": "ppo",
        "env_id": "SafetyPointGoal2-v0",
        "timesteps": 2_500_000,
        "seed": 0,
        "n_envs": 8,
    },

    "ppo_3m500k": {
        "backend": "sb3",
        "algorithm": "ppo",
        "env_id": "SafetyPointGoal2-v0",
        "timesteps": 3_500_000,
        "seed": 0,
        "n_envs": 8,
    },

    "ppo_500k": {
        "backend": "sb3",
        "algorithm": "ppo",
        "env_id": "SafetyPointGoal2-v0",
        "timesteps": 500_000,
        "seed": 0,
        "n_envs": 8,
    },

    "ppo_4m500k_goal2_safe": {
        "backend": "sb3",
        "algorithm": "ppo",
        "env_id": "SafetyPointGoal2-v0",
        "timesteps": 4_500_000,
        "seed": 0,
        "n_envs": 8,
    },

    "ppo_5m_goal2_safe": {
        "backend": "sb3",
        "algorithm": "ppo",
        "env_id": "SafetyPointGoal2-v0",
        "timesteps": 5_000_000,
        "seed": 0,
        "n_envs": 8,
    },

    "ppo_5m500k_goal2_safe_k15": {
        "backend": "sb3",
        "algorithm": "ppo",
        "env_id": "SafetyPointGoal2-v0",
        "timesteps": 5_500_000,
        "seed": 0,
        "n_envs": 8,
    },

    "ppo_5m550k_goal2_safe_vase_k15": {
        "backend": "sb3",
        "algorithm": "ppo",
        "env_id": "SafetyPointGoal2-v0",
        "timesteps": 5_550_000,
        "seed": 0,
        "n_envs": 8,
    },

    "ppo_5m600k_goal2_safe_vase_k15": {
        "backend": "sb3",
        "algorithm": "ppo",
        "env_id": "SafetyPointGoal2-v0",
        "timesteps": 5_600_000,
        "seed": 0,
        "n_envs": 8,
    },

    "ppo_5m700k_goal2": {
        "backend": "sb3",
        "algorithm": "ppo",
        "env_id": "SafetyPointGoal2-v0",
        "timesteps": 5_700_000,
        "seed": 0,
        "n_envs": 8,
    },

    "ppo_6m_goal2": {
        "backend": "sb3",
        "algorithm": "ppo",
        "env_id": "SafetyPointGoal2-v0",
        "timesteps": 6_000_000,
        "seed": 0,
        "n_envs": 8,
    },

    "ppolag_5k": {
        "backend": "omnisafe",
        "algorithm": "ppolag",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 5000,
        "save_video": True,
        "video_length": 1000,
    },

    "ppolag_15k": {
        "backend": "omnisafe",
        "algorithm": "ppolag",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 15_000,
        "save_video": True,
        "video_length": 1000,
    },

    "ppolag_20k": {
        "backend": "omnisafe",
        "algorithm": "ppolag",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 20000,
        "save_video": True,
        "video_length": 1000,
    },

    "ppolag_100k": {
        "backend": "omnisafe",
        "algorithm": "ppolag",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 100_000,
        "save_video": True,
        "video_length": 1000,
    },

    "ppolag_500k": {
        "backend": "omnisafe",
        "algorithm": "ppolag",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 500_000,
        "save_video": True,
        "video_length": 1000,
    },

    "ppolag_1m500k": {
        "backend": "omnisafe",
        "algorithm": "ppolag",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 1_500_000,
        "save_video": True,
        "video_length": 2000,
    },

    "ppolag_3m500k": {
        "backend": "omnisafe",
        "algorithm": "ppolag",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 3_500_000,
        "save_video": True,
        "video_length": 2000,
    }
}