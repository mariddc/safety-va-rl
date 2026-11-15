# Frozen PPO hyperparameters for fine-tunes, one dict per base, shared by policy_ft and policy_ref.
# A fine-tune definition spreads **FINETUNE_HPARAMS[base] and sets only beta and seed itself.
FINETUNE_HPARAMS = {
    "ppo_5m_goal2_safe": {
        "learning_rate": 1e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "n_envs": 8,
        "timesteps": 500_000,
        "device": "cpu",
    },
    "ppo_5m700k_goal2": {
        "learning_rate": 1e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "n_envs": 8,
        "timesteps": 500_000,
        "device": "cpu",
    },
}
