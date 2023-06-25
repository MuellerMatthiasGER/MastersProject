import random
from minigrid.core.constants import COLOR_NAMES
from minigrid.envs.babyai.core.roomgrid_level import RejectSampling, RoomGridLevel
from minigrid.envs.babyai.core.verifier import GoToInstr, ObjDesc

from minigrid.manual_control import ManualControl

from gymnasium.envs.registration import register

class CustomLevel(RoomGridLevel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def seed(self, random_seed):
        pass


class LearnRed(CustomLevel):
    """

    ## Description

    Go to the red ball, single room, with distractors.
    There is only only red ball.
    The distractors are all balls such that the agent must learn to distinguish red.
    This level has distractors but doesn't make use of language.

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
    | 5   | toggle       | Unused            |
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

    1. The agent goes to the red ball.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - ``
    
    """

    def __init__(self, room_size=8, num_dists=4, **kwargs):
        self.num_dists = num_dists
        super().__init__(num_rows=1, num_cols=1, room_size=room_size, **kwargs)

    def gen_mission(self):
        self.place_agent()
        obj, _ = self.add_object(0, 0, "ball", "red")

        colors = COLOR_NAMES.copy()
        colors.remove('red')
        for _ in range(self.num_dists):
            color = self._rand_elem(colors)
            self.add_object(0, 0, kind='ball', color=color)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class LearnBall(CustomLevel):
    """

    ## Description

    Go to the red ball, single room, with distractors.
    There is only only red ball.
    The distractors are all red such that the agent must learn to distinguish a ball.
    This level has distractors but doesn't make use of language.

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
    | 5   | toggle       | Unused            |
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

    1. The agent goes to the red ball.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - ``

    """

    def __init__(self, room_size=8, num_dists=4, **kwargs):
        self.num_dists = num_dists
        super().__init__(num_rows=1, num_cols=1, room_size=room_size, **kwargs)

    def gen_mission(self):
        self.place_agent()
        obj, _ = self.add_object(0, 0, "ball", "red")

        kinds = ['key', 'box']
        for _ in range(self.num_dists):
            kind = self._rand_elem(kinds)
            self.add_object(0, 0, kind=kind, color='red')

        # Make sure no unblocking is required
        self.check_objs_reachable()

        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class FindRedBall(CustomLevel):
    """

    ## Description

    Go to the red ball, single room, with distractors.
    There is only only red ball.
    This level has distractors but doesn't make use of language.

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
    | 5   | toggle       | Unused            |
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

    1. The agent goes to the red ball.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - ``

    """

    def __init__(self, room_size=8, num_dists=4, **kwargs):
        self.num_dists = num_dists
        super().__init__(num_rows=1, num_cols=1, room_size=room_size, **kwargs)

    def gen_mission(self):
        self.place_agent()
        obj, _ = self.add_object(0, 0, "ball", "red")

        colors = COLOR_NAMES.copy()
        colors_no_red = COLOR_NAMES.copy()
        colors_no_red.remove('red')
        kinds = ['ball', 'key', 'box']
        for _ in range(self.num_dists):
            kind = random.choice(kinds)
            if kind == 'ball':
                color = self._rand_elem(colors_no_red)
            else:
                color = self._rand_elem(colors)
            self.add_object(0, 0, kind=kind, color=color)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


register(
    id="MyMiniGrid-LearnRed-v0",
    entry_point="my_minigrid:LearnRed",
)

register(
    id="MyMiniGrid-LearnBall-v0",
    entry_point="my_minigrid:LearnBall",
)

register(
    id="MyMiniGrid-FindRedBall-v0",
    entry_point="my_minigrid:FindRedBall",
)


if __name__ == '__main__':
    env = LearnBall(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env)
    manual_control.start()