import gym
import xmagical

# This must be called before making any Gym envs.
xmagical.register_envs()

# List all available environments.
print(xmagical.ALL_REGISTERED_ENVS)

# Create a demo variant for the SweepToTop task with a gripper agent.
env = gym.make('SweepToTop-Gripper-Pixels-Allo-Demo-v0')
obs = env.reset()
print(obs.shape)  # (384, 384, 3)
env.render(mode='human')
env.close()

# Now create a test variant of this task with a shortstick agent,
# an egocentric view and a state-based observation space.
env = gym.make('SweepToTop-Shortstick-State-Ego-TestLayout-v0')
init_obs = env.reset()
print(init_obs.shape)  # (16,)
env.close()
