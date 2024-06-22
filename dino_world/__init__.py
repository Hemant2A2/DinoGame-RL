from gym.envs.registration import register

register(
    id="dino_world/DinoWorld-v0",
    entry_point="dino_world.envs:DinoGameEnv",
)
