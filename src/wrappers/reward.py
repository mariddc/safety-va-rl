from src.wrappers.SafetyGymSB3Wrapper import SafetyGymSB3Wrapper
from src.wrappers.FastSafeRewardWrapper import FastSafeRewardWrapper
from src.wrappers.FastSafeCompleteRewardWrapper import FastSafeCompleteRewardWrapper

# Keys used in experiment definitions ("reward_wrapper")
REWARD_WRAPPERS = {
    "native": SafetyGymSB3Wrapper,           # env reward, no shaping, no goal termination
    "fastsafe": FastSafeRewardWrapper,
    "complete": FastSafeCompleteRewardWrapper,
}


def wrap_reward(env, exp):
    return REWARD_WRAPPERS[exp["reward_wrapper"]](env, **exp["reward_kwargs"])


def reward_config(exp):
    return {"reward_wrapper": exp["reward_wrapper"], "reward_kwargs": exp["reward_kwargs"]}
