from abc import ABC
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Key
from minigrid.minigrid_env import MiniGridEnv

from minigrid.manual_control import ManualControl

from gymnasium.envs.registration import register

import math


class CustomLevel(MiniGridEnv, ABC):
    def __init__(self, width, height, numObjs=4, max_steps=None, **kwargs):
        self.numObjs = numObjs
        self.width = width
        self.height = height
        # Types of objects to be generated
        self.obj_types = ["key", "ball", "box"]

        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES, self.obj_types],
        )

        if max_steps is None:
            max_steps = width * height

        super().__init__(
            mission_space=mission_space,
            width=self.width,
            height=self.height,
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

        self.mission = "go to the red ball"
        # print(self.mission)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # ax, ay = self.agent_pos
        # tx, ty = self.target_pos

        # Toggle/pickup action terminates the episode
        # if action == self.actions.toggle:
        #     terminated = True

        # Reward performing the done action next to the target object
        # if action == self.actions.done:
        #     if (ax == tx and abs(ay - ty) == 1) or (ay == ty and abs(ax - tx) == 1):
        #         reward = self._reward()
        #     terminated = True


        ax, ay = self.front_pos
        tx, ty = self.target_pos

        # Terminate if agent picks up an object
        if action == self.actions.pickup:
            terminated = True
            # Reward if agent is in front of target
            if (ax == tx and ay == ty):
                reward = self._reward()
            else:
                reward = 0

        return obs, reward, terminated, truncated, info
    
    def create_object(self, colors, types, blacklist=None, top=None, size=None):
        if isinstance(colors, list):
            objColor = self._rand_elem(colors)
        else:
            objColor = colors
        
        if isinstance(types, list):
            # ensure that no object is created that is listed in the blacklist as a tuple (color, type)
            if blacklist:
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
        
        pos = self.place_obj(obj, top=top, size=size)
        return obj, pos
    
    def seed(self, random_seed):
        pass

class CustomLevelMini(CustomLevel, ABC):
    def __init__(self, width=7, height=5, numObjs=1, **kwargs):
        super().__init__(width, height, numObjs, **kwargs)

    def _gen_grid(self, width, height, obj_colors, obj_types, blacklist=None):
        super()._gen_grid(width, height)

        left_obj_top_pos = (1, 1)
        right_obj_top_pos = (math.ceil((width + 1) / 2) , 1)
        obj_place_size = ((math.floor(width - 3) / 2), height - 2)

        agent_top_pos = (math.floor((width - 1) / 2), 1)
        agent_place_size = (2 - (width % 2), height - 2)

        # Generate the red ball on one side and a distractor on the other side
        if self._rand_bool():
            self.target_pos = self.place_obj(Ball('red'), top=left_obj_top_pos, size=obj_place_size)
            self.create_object(colors=obj_colors, types=obj_types, blacklist=blacklist, top=right_obj_top_pos, size=obj_place_size)
        else:
            self.target_pos = self.place_obj(Ball('red'), top=right_obj_top_pos, size=obj_place_size)
            self.create_object(colors=obj_colors, types=obj_types, blacklist=blacklist, top=left_obj_top_pos, size=obj_place_size)

        # Place agent
        self.place_agent(top=agent_top_pos, size=agent_place_size)

class CustumLevelBigSquare(CustomLevel):
    def __init__(self, size=6, **kwargs):
        super().__init__(width=size, height=size, numObjs=4, **kwargs)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Generate the red ball
        _, self.target_pos = self.create_object('red', 'ball')

        # Randomize the agent start position and orientation
        self.place_agent()


class LearnRedMini(CustomLevelMini):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _gen_grid(self, width, height):
        # Types and colors the distractor can have
        colors = COLOR_NAMES.copy()
        colors.remove('red')

        super()._gen_grid(width, height, obj_colors=colors, obj_types='ball')

class LearnRed(CustumLevelBigSquare):
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
    

class LearnBallMini(CustomLevelMini):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _gen_grid(self, width, height):
        # Types and colors the distractor can have
        types = self.obj_types.copy()
        types.remove('ball')

        super()._gen_grid(width, height, obj_colors='red', obj_types=types)

class LearnBall(CustumLevelBigSquare):
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


class FindRedBallMini(CustomLevelMini):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _gen_grid(self, width, height):
        # Types and colors the distractor can have
        colors = COLOR_NAMES.copy()
        types = self.obj_types

        super()._gen_grid(width, height, obj_colors=colors, obj_types=types, blacklist=[('red', 'ball')])

class FindRedBall(CustumLevelBigSquare):
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
        types = self.obj_types

        # Generate distractors
        for _ in range(self.numObjs - 1):
            self.create_object(colors, types, blacklist=[('red', 'ball')])


register(
    id="MyMiniGrid-LearnRedMini-v1",
    entry_point="my_minigrid_new:LearnRedMini",
)

register(
    id="MyMiniGrid-LearnBallMini-v1",
    entry_point="my_minigrid_new:LearnBallMini",
)

register(
    id="MyMiniGrid-FindRedBallMini-v1",
    entry_point="my_minigrid_new:FindRedBallMini",
)

register(
    id="MyMiniGrid-LearnRed-v1",
    entry_point="my_minigrid_new:LearnRed",
)

register(
    id="MyMiniGrid-LearnBall-v1",
    entry_point="my_minigrid_new:LearnBall",
)

register(
    id="MyMiniGrid-FindRedBall-v1",
    entry_point="my_minigrid_new:FindRedBall",
)


if __name__ == '__main__':
    env = LearnRedMini(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env)
    manual_control.start()