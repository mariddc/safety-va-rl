FINETUNES = {
    "ppo_5m_goal2_safe_qft_v1": {
        "name": "ppo_5m_goal2_safe_qft_v1",
        "backend": "sb3",
        "algorithm": "ppo",
        "env_id": "SafetyPointGoal2-v0",

        # fine-tune
        "base_exp": "ppo_5m_goal2_safe",   # load model from this 
        "timesteps": 500_000,
        "seed": 0,
        "n_envs": 8,

        # q penalty
        "q_path": "logs/datasets/eil/ppo/SafetyPointGoal2-v0/ppo_5m_goal2_safe/SafetyPointGoal2-v0/eil_aL0.2_aE0.0_v2/q_theta.pt",
        "beta": 1.0,
        "B": 0.0,
        "q_device": "cpu",

        # PPO fine-tune
        "learning_rate": 1e-4,
        "n_steps": 2048,
        "batch_size": 2048,
        "gamma": 0.99,
        "clip_range": 0.2,

    },

    "ppo_5m_goal2_safe_qft_v2": {
        "name": "ppo_5m_goal2_safe_qft_v2",
        "backend": "sb3",
        "algorithm": "ppo",
        "env_id": "SafetyPointGoal2-v0",

        "base_exp": "ppo_5m_goal2_safe", 
        "timesteps": 500_000,
        "seed": 0,
        "n_envs": 8,

        "q_path": "logs/datasets/eil/ppo/SafetyPointGoal2-v0/ppo_5m_goal2_safe/SafetyPointGoal2-v0/eil_aL0.2_aE0.0_v2/q_theta.pt",
        "beta": 0.05,
        "B": 0.0,
        "q_device": "cpu",

    },

    "ppo_5m_goal2_safe_qft_v3": {
        "name": "ppo_5m_goal2_safe_qft_v3",
        "backend": "sb3",
        "algorithm": "ppo",
        "env_id": "SafetyPointGoal2-v0",

        "base_exp": "ppo_5m_goal2_safe", 
        "timesteps": 200_000,
        "seed": 0,
        "n_envs": 8,

        "q_path": "logs/datasets/eil/ppo/SafetyPointGoal2-v0/ppo_5m_goal2_safe/SafetyPointGoal2-v0/eil_aL0.2_aE0.0_v2/q_theta.pt",
        "beta": 0.05,
        "B": 0.0,
        "q_device": "cpu",

        "learning_rate": 1e-4,

    },

     "ppo_5m_goal2_safe_qft_teste": {
        "name": "ppo_5m_goal2_safe_qft_teste",
        "backend": "sb3",
        "algorithm": "ppo",
        "env_id": "SafetyPointGoal2-v0",

        "base_exp": "ppo_5m_goal2_safe", 
        "timesteps": 20_000,
        "seed": 0,
        "n_envs": 8,

        "q_path": "logs/datasets/eil/ppo/SafetyPointGoal2-v0/ppo_5m_goal2_safe/SafetyPointGoal2-v0/eil_aL0.2_aE0.0_v2/q_theta.pt",
        "beta": 0.05,
        "B": -0.005,
        "q_device": "cpu",

        "learning_rate": 1e-4,

    },

    "ppo_5m_goal2_inverted_reward": {
        "name": "ppo_5m_goal2_inverted_reward",
        "backend": "sb3",
        "algorithm": "ppo",
        "env_id": "SafetyPointGoal2-v0",

        "base_exp": "ppo_5m_goal2_safe", 
        "timesteps": 500_000,
        "seed": 0,
        "n_envs": 8,

        "q_path": "logs/datasets/eil/ppo/SafetyPointGoal2-v0/ppo_5m_goal2_safe/SafetyPointGoal2-v0/eil_aL0.2_aE0.0_v2/q_theta.pt",
        "beta": 0.05,
        "B": -0.005,
        "q_device": "cpu",

        "learning_rate": 1e-4,

    },

    "ppo_5m700k_goal2_inverted": {
        "name": "ppo_5m700k_goal2_inverted",
        "backend": "sb3",
        "algorithm": "ppo",
        "env_id": "SafetyPointGoal2-v0",

        "base_exp": "ppo_5m700k_goal2", 
        "timesteps": 500_000,
        "seed": 0,
        "n_envs": 8,

        "q_path": "logs/datasets/eil/ppo/SafetyPointGoal2-v0/ppo_5m700k_goal2/SafetyPointGoal2-v0/eil_aL0.2_aE0.0_v1/q_theta.pt",
        "beta": 0.5,
        "B": -0.02,
        "q_device": "cpu",

        "learning_rate": 1e-4,

    },
} 