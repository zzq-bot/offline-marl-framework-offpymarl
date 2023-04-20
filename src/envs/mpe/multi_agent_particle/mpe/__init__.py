from gym.envs.registration import register
import mpe.scenarios as scenarios

def _register(scenario_name, gymkey):
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    register(
        gymkey,
        entry_point="mpe.environment:MultiAgentEnv",
        kwargs={
            "world": world,
            "reset_callback": scenario.reset_world,
            "reward_callback": scenario.reward,
            "observation_callback": scenario.observation,

            "done_callback": scenario.done,
        },
    )

scenario_name = "simple_tag"
gymkey = "SimpleTag-heuristic-v0"
_register(scenario_name, gymkey)

scenario_name = "simple_tag_fix"
gymkey = "SimpleTag-fix-frozen-v0"
_register(scenario_name, gymkey)

scenario_name = "simple_tag_bull"
gymkey = "SimpleTag-bull-bull-v0"
_register(scenario_name, gymkey)

scenario_name = "simple_tag_flash"
gymkey = "SimpleTag-flash-random-v0"
_register(scenario_name, gymkey)

for i in range(1, 6):
    scenario_name = f"simple_tag{i}"
    gymkey = f"SimpleTag-heuristic{i}-v0"
    _register(scenario_name, gymkey)

    scenario_name = f"simple_tag_fix{i}"
    gymkey = f"SimpleTag-fix-frozen{i}-v0"
    _register(scenario_name, gymkey)

    scenario_name = f"simple_tag_bull{i}"
    gymkey = f"SimpleTag-bull-bull{i}-v0"
    _register(scenario_name, gymkey)

    scenario_name = f"simple_tag_flash{i}"
    gymkey = f"SimpleTag-flash-random{i}-v0"
    _register(scenario_name, gymkey)
