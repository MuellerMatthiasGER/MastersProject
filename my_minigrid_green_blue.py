from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import ObjDesc, PickupInstr

from minigrid.core.world_object import Ball, Box

from minigrid.manual_control import ManualControl

from gymnasium.envs.registration import register


class Pickup(RoomGridLevel):
    def __init__(self, target_idx=None, room_size=6, agent_view_size=3, **kwargs):
        super().__init__(
            num_rows=1, num_cols=1, 
            room_size=room_size, 
            max_steps=room_size**2, 
            agent_view_size=agent_view_size, 
            **kwargs)
        self.objs = [Ball('green'), Ball('blue'), Box('green'), Box('blue')]
        self.target_idx = target_idx

    def gen_mission(self):
        self.place_agent()
        self.connect_all()

        num_objs = len(self.objs)

        if self.target_idx is not None:
            target_idx = self.target_idx
        else:
            target_idx = self._rand_int(0, num_objs)

        for i in range(num_objs):
            idx = (target_idx + i) % num_objs
            self.place_in_room(0, 0, self.objs[idx])

        self.check_objs_reachable()
        target = self.objs[target_idx]
        self.instrs = PickupInstr(ObjDesc(target.type, target.color), strict=True)

    def seed(self, random_seed):
        pass

register(
    id="MyMiniGrid-PickupGreenBall-v0",
    entry_point="my_minigrid_green_blue:Pickup",
    kwargs={"target_idx": 0},
)

register(
    id="MyMiniGrid-PickupBlueBall-v0",
    entry_point="my_minigrid_green_blue:Pickup",
    kwargs={"target_idx": 1},
)

register(
    id="MyMiniGrid-PickupGreenBox-v0",
    entry_point="my_minigrid_green_blue:Pickup",
    kwargs={"target_idx": 2},
)

register(
    id="MyMiniGrid-PickupBlueBox-v0",
    entry_point="my_minigrid_green_blue:Pickup",
    kwargs={"target_idx": 3},
)

register(
    id="MyMiniGrid-PickupGreenBlue-v0",
    entry_point="my_minigrid_green_blue:Pickup",
)


if __name__ == '__main__':
    env = Pickup(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env)
    manual_control.start()