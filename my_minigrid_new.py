from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Key
from minigrid.minigrid_env import MiniGridEnv


class CustomLevel(MiniGridEnv):
    def __init__(self, size=6, numObjs=4, max_steps = None, **kwargs):
        self.numObjs = numObjs
        self.size = size
        # Types of objects to be generated
        self.obj_types = ["key", "ball", "box"]

        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES, self.obj_types],
        )

        if max_steps is None:
            max_steps = 5 * size**2

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission(color: str, obj_type: str):
        return "go to the red ball"
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate the red ball
        _, self.target_pos = self.create_object('red', 'ball')

        # Randomize the agent start position and orientation
        self.place_agent()

        self.mission = "go to the red ball"
        # print(self.mission)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        ax, ay = self.agent_pos
        tx, ty = self.target_pos

        # Toggle/pickup action terminates the episode
        if action == self.actions.toggle:
            terminated = True

        # Reward performing the done action next to the target object
        if action == self.actions.done:
            if (ax == tx and abs(ay - ty) == 1) or (ay == ty and abs(ax - tx) == 1):
                reward = self._reward()
            terminated = True

        return obs, reward, terminated, truncated, info

    def create_object(self, colors, types, blacklist=None):
        if isinstance(colors, list):
            objColor = self._rand_elem(colors)
        else:
            objColor = colors
        
        if isinstance(types, list):
            # ensure that no object is created that is listed in the blacklist as a tuple (color, type)
            if blacklist:
                types = types.copy()
                for col, typ in blacklist:
                    if objColor == col and typ in types:
                        types.remove(typ)
            
            objType = self._rand_elem(types)
        else:
            objType = types

        if objType == "key":
            obj = Key(objColor)
        elif objType == "ball":
            obj = Ball(objColor)
        elif objType == "box":
            obj = Box(objColor)
        else:
            raise ValueError(
                "{} object type given. Object type can only be of values key, ball and box.".format(objType)
            )
        
        pos = self.place_obj(obj)
        return obj, pos
    
    def seed(self, random_seed):
        pass



class LearnRed(CustomLevel):
    """

    ## Description

    TODO

    ## Mission Space

    "go to the red ball"

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Unused            |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Toggle            |
    | 6   | done         | Unused            |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent goes to the red ball and toggles.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - ``
    
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Types and colors of distractors we can generate
        colors = COLOR_NAMES.copy()
        colors.remove('red')

        # Generate distractors
        for _ in range(self.numObjs - 1):
            self.create_object(colors, 'ball')
    

class LearnBall(CustomLevel):
    """

    ## Description

    TODO

    ## Mission Space

    "go to the red ball"

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Unused            |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Toggle            |
    | 6   | done         | Unused            |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent goes to the red ball and toggles.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - ``
    
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Types and colors of distractors we can generate
        types = self.obj_types.copy()
        types.remove('ball')

        # Generate distractors
        for _ in range(self.numObjs - 1):
            self.create_object('red', types)


class FindRedBall(CustomLevel):
    """

    ## Description

    TODO

    ## Mission Space

    "go to the red ball"

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Unused            |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Toggle            |
    | 6   | done         | Unused            |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent goes to the red ball and toggles.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - ``
    
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Generate distractors
        for _ in range(self.numObjs - 1):
            self.create_object(COLOR_NAMES, self.obj_types, blacklist=[('red', 'ball')])


# from gymnasium.envs.registration import register
# register("MyMiniGrid-LearnRed-v0", "my_minigrid_new:LearnRed")
# register("MyMiniGrid-LearnBall-v0", "my_minigrid_new:LearnBall")
# register("MyMiniGrid-FindRedBall-v0", "my_minigrid_new:FindRedBall")


# if __name__ == '__main__':
    # env = LearnBall(render_mode="human")

    # # enable manual control for testing
    # manual_control = ManualControl(env)
    # manual_control.start()