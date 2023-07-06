from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import ObjDesc, PickupInstr

from minigrid.core.world_object import Ball, Box

from minigrid.manual_control import ManualControl

from gymnasium.envs.registration import register


class Pickup(RoomGridLevel):
    def __init__(self, target_color=None, target_type=None, room_size=7, agent_view_size=5, **kwargs):
        super().__init__(
            num_rows=1, num_cols=1, 
            room_size=room_size, 
            max_steps=room_size**2, 
            agent_view_size=agent_view_size, 
            **kwargs)
        self.objs = [Ball('green'), Ball('blue'), Box('green'), Box('blue')]
        self.target_color = target_color
        self.target_type = target_type

        if target_type == 'ball':
            if target_color == 'blue':
                self.start_idx = 1
            else:
                self.start_idx = 0
        else:
            if target_color == 'blue':
                self.start_idx = 3
            else:
                self.start_idx = 2

    def gen_mission(self):
        self.place_agent()
        self.connect_all()

        num_objs = len(self.objs)

        for i in range(num_objs):
            idx = (self.start_idx + i) % num_objs
            self.place_in_room(0, 0, self.objs[idx])

        self.check_objs_reachable()
        self.instrs = PickupInstr(ObjDesc(self.target_type, self.target_color), strict=True)

    def seed(self, random_seed):
        pass


register(
    id="MyMiniGrid-Pickup-v0",
    entry_point="my_minigrid_green_blue:Pickup",
    kwargs={"target_color": None, "target_type": None},
)

register(
    id="MyMiniGrid-PickupGreen-v0",
    entry_point="my_minigrid_green_blue:Pickup",
    kwargs={"target_color": "green", "target_type": None},
)

register(
    id="MyMiniGrid-PickupBlue-v0",
    entry_point="my_minigrid_green_blue:Pickup",
    kwargs={"target_color": "blue", "target_type": None},
)

register(
    id="MyMiniGrid-PickupBall-v0",
    entry_point="my_minigrid_green_blue:Pickup",
    kwargs={"target_color": None, "target_type": "ball"},
)

register(
    id="MyMiniGrid-PickupBox-v0",
    entry_point="my_minigrid_green_blue:Pickup",
    kwargs={"target_color": None, "target_type": "box"},
)

register(
    id="MyMiniGrid-PickupGreenBall-v0",
    entry_point="my_minigrid_green_blue:Pickup",
    kwargs={"target_color": "green", "target_type": "ball"},
)

register(
    id="MyMiniGrid-PickupBlueBall-v0",
    entry_point="my_minigrid_green_blue:Pickup",
    kwargs={"target_color": "blue", "target_type": "ball"},
)

register(
    id="MyMiniGrid-PickupGreenBox-v0",
    entry_point="my_minigrid_green_blue:Pickup",
    kwargs={"target_color": "green", "target_type": "box"},
)

register(
    id="MyMiniGrid-PickupBlueBox-v0",
    entry_point="my_minigrid_green_blue:Pickup",
    kwargs={"target_color": "blue", "target_type": "box"},
)


if __name__ == '__main__':
    env = Pickup(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env)
    manual_control.start()