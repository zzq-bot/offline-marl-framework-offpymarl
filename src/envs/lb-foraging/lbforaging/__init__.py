from gym.envs.registration import registry, register, make, spec
from itertools import product

#sizes = range(4, 20)
#players = range(2, 20)
#foods = range(1, 10)
#coop = [True, False]
#_grid_obs = [True, False]

sizes = range(4, 8)
players = [2, 3]
foods = [1]
coop = [True, False]
_grid_obs = [False]
food_pos = ["topright", "right", "bottomright", "bottom", "bottomleft",
            "topright1", "right1", "bottomright1", "bottom1", "bottomleft1",
            "topright2", "right2", "bottomright2", "bottom2", "bottomleft2",
            "topright3", "right3", "bottomright3", "bottom3", "bottomleft3",
            "right4", "bottom4",
            "right5", "bottom5",]

for s, p, f, c, grid, pos in product(sizes, players, foods, coop, _grid_obs, food_pos):
    for sight in range(1, s + 1):
        register(
            id="Foraging{4}-{0}x{0}-{1}p-{2}f{3}{5}-{6}-v1".format(s, p, f, "-coop" if c else "", "" if sight == s else f"-{sight}s", "-grid" if grid else "", pos),
            entry_point="lbforaging.foraging:ForagingEnv",
            kwargs={
                "players": p,
                "max_player_level": 3,
                "field_size": (s, s),
                "max_food": f,
                "sight": sight,
                "max_episode_steps": 500,
                "force_coop": c,
                "food_pos": pos,
                "_grid_observation": grid,
            },
        )
