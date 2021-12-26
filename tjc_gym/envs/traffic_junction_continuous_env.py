import copy
import math

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

ENV_WIDTH = 1
GRASS_WIDTH = ENV_WIDTH * 0.425
CAR_WIDTH = ENV_WIDTH * 0.0375
CAR_LENGTH = 2 * CAR_WIDTH


class Point(object):
    def __init__(self, xcoord=0, ycoord=0) -> None:
        self.x = xcoord
        self.y = ycoord


class EntityState(object):
    def __init__(self) -> None:
        self.position = None


class AgentState(EntityState):
    def __init__(self) -> None:
        super(AgentState, self).__init__()
        self.direction = None  # vector e.g up: (0,1)
        self.route = None  # 1, 2, or 3 (straigt, turn left, turn right)
        self.step_count = 0
        self.on_the_road = False
        self.done = False
        self.colliding = (False, None)  # (colliding, whom). Wether the agent was colliding in last step and with whom


class Entity(object):
    def __init__(self) -> None:
        self.state = EntityState()


class Action(object):
    def __init__(self) -> None:
        self.movement = None


class Agent(Entity):
    def __init__(self, index) -> None:
        super().__init__()
        self.index = index
        self.action = Action()
        self.state = AgentState()

    def get_rect_coords(self):
        # return coordinates of the agent's top right corner and bottom left corner
        top_right, bottom_left = None, None

        # translate so bottom left is (0,0)
        x, y = self.state.position[0] + 1, self.state.position[1] + 1

        d = self.state.direction

        # find rectangle corners depending on the agent's direction
        if d == (0, 1):
            top_right = Point(x + CAR_WIDTH / 2, y)
            bottom_left = Point(x - CAR_WIDTH / 2, y - CAR_LENGTH)
        elif d == (0, -1):
            top_right = Point(x + CAR_WIDTH / 2, y + CAR_LENGTH)
            bottom_left = Point(x - CAR_WIDTH / 2, y)
        elif d == (1, 0):
            top_right = Point(x, y + CAR_WIDTH / 2)
            bottom_left = Point(x - CAR_LENGTH, y - CAR_WIDTH / 2)
        elif d == (-1, 0):
            top_right = Point(x + CAR_LENGTH, y + CAR_WIDTH / 2)
            bottom_left = Point(x, y - CAR_WIDTH / 2)

        return top_right, bottom_left

    def done(self):
        self.state.done = True
        # self.state.on_the_road = False
        self.state.position = (-1, -1)


class TrafficJunctionContinuousEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        n_max=4,
        max_steps=500,
        arrive_prob=0.05,
        r_fov=3,
        step_cost=-0.01,
        collision_cost=-100,
        movement_scale_factor=0.01,
        observability="global",
        reward_callback=lambda rewards, _: rewards,
    ) -> None:
        self.seed()

        self.n_agents = n_max
        self.curr_cars_count = 0
        self.max_steps = max_steps
        self.arrive_prob = arrive_prob
        self.collision_cost = collision_cost
        self.step_cost = step_cost
        self.movement_scale_factor = movement_scale_factor
        self.observability = observability

        # initalize field of view
        if observability == "fov":
            self.set_r_fov(r_fov)

        self.reward_callback = reward_callback

        self._agents = self.n_agents * [None]
        self._n_routes = 1  # only possible to move forward (no turning)

        self._entry_gates = {
            "top": (GRASS_WIDTH + CAR_WIDTH, ENV_WIDTH),
            "right": (ENV_WIDTH, ENV_WIDTH - GRASS_WIDTH - CAR_WIDTH),
            "bottom": (ENV_WIDTH - GRASS_WIDTH - CAR_WIDTH, 0),
            "left": (0, GRASS_WIDTH + CAR_WIDTH),
        }

        self._directions = {"down": (0, -1), "left": (-1, 0), "up": (0, 1), "right": (1, 0)}
        self._n_directions = len(self._directions)

        # destinations: { direction : destination coordinates }
        self._destinations = {}
        dest_positions = [
            (GRASS_WIDTH + CAR_WIDTH, 0),  # bottom
            (0, GRASS_WIDTH + CAR_WIDTH),  # left
            (ENV_WIDTH - GRASS_WIDTH - CAR_WIDTH, ENV_WIDTH),  # top
            (ENV_WIDTH, GRASS_WIDTH + CAR_WIDTH),  # right
        ]
        for i, direction in enumerate(self._directions.values()):
            self._destinations[direction] = dest_positions[i]

        # dict: { starting_place: direction_vector }
        self._route_vectors = {
            self._entry_gates["top"]: self._directions["down"],  # at top moving down
            self._entry_gates["right"]: self._directions["left"],  # at right moving left
            self._entry_gates["bottom"]: self._directions["up"],  # at bottom moving up
            self._entry_gates["left"]: self._directions["right"],  # at left moving right
        }

        # action and observation space
        self.action_space = []
        for i in range(self.n_agents):
            self.action_space.append(spaces.Box(low=0, high=1, shape=(1,)))

        self._init_obs_space()

        # env info
        self._step_count = 0
        self._acc_unique_collisions = 0
        self._avg_speed = 0
        self._total_reward = 0

        # render
        self._viewer = None

    def reset(self):
        self._agents = self.n_agents * [None]

        # env info
        self._step_count = 0
        self._acc_unique_collisions = 0
        self._avg_speed = 0
        self._total_reward = 0

        self._reset_render()
        self._reset_environment()

        return self.get_agent_obs()

    # https://github.com/openai/gym/blob/v0.21.0/gym/envs/classic_control/cartpole.py#L99
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_r_fov(self, r_fov):
        self._r_fov = r_fov
        self._fov_w = 2 * self._r_fov + 1
        self._fov_h = 2 * self._r_fov + 1

    def _is_region_on_road(self, ll, tr, scale):
        grass_width = GRASS_WIDTH * scale
        env_width = ENV_WIDTH * scale

        def is_point_in_grass(x, y):
            if x <= grass_width:
                if y <= grass_width:  # lower left
                    return True
                elif y >= env_width - grass_width:  # top left
                    return True
            elif x >= env_width - grass_width:
                if y <= grass_width:  # lower right
                    return True
                elif y >= env_width - grass_width:  # top right
                    return True
            return False

        return is_point_in_grass(ll[0], ll[1]) and is_point_in_grass(tr[0], tr[1])

    def _init_obs_space(self):
        self.observation_space = []

        if self.observability == "fov":
            for i in range(self.n_agents):
                self.observation_space.append(
                    spaces.Box(
                        low=-math.pi,
                        high=math.pi,
                        shape=(self._fov_w, self._fov_h, len(self._directions) + 2),
                        dtype=float,
                    )
                )
        elif self.observability == "global":
            for i in range(self.n_agents):
                self.observation_space.append(
                    spaces.Box(
                        low=-math.pi,
                        high=math.pi,
                        shape=(self.n_agents, len(self._directions) + 3),
                        dtype=float,
                    )
                )

    def _get_fov(self, agent):
        fov = np.full((self._fov_w, self._fov_h, 6), 0.0, dtype=np.float32)
        others_in_vision = np.zeros(self.n_agents, dtype=int)

        if not agent.state.on_the_road:
            return fov, others_in_vision

        ego_x, ego_y = self._r_fov, self._r_fov

        scale = 10000
        car_w = int(CAR_WIDTH * scale)
        car_l = int(CAR_LENGTH * scale)

        p = (int(agent.state.position[0] * scale), int(agent.state.position[1] * scale))
        d = agent.state.direction

        a_region_ll = None
        x_inc = None
        y_inc = None
        inside_fn = None
        if d == self._directions["right"]:
            a_region_ll = (p[0] - car_l, p[1] + car_w)
            x_inc, y_inc = 1, -1
            inside_fn = lambda ll, tr, pos: (ll[0] < pos[0] <= tr[0] and ll[1] > pos[1] >= tr[1])
        elif d == self._directions["left"]:
            a_region_ll = (p[0] + car_l, p[1] - car_w)
            x_inc, y_inc = -1, 1
            inside_fn = lambda ll, tr, pos: (ll[0] >= pos[0] > tr[0] and ll[1] <= pos[1] < tr[1])
        elif d == self._directions["up"]:
            a_region_ll = (p[0] - car_w, p[1] - car_l)
            x_inc, y_inc = 1, 1
            inside_fn = lambda ll, tr, pos: (ll[0] < pos[0] <= tr[0] and ll[1] < pos[1] <= tr[1])
        elif d == self._directions["down"]:
            a_region_ll = (p[0] + car_w, p[1] + car_l)
            x_inc, y_inc = -1, -1
            inside_fn = lambda ll, tr, pos: (ll[0] > pos[0] >= tr[0] and ll[1] > pos[1] >= tr[1])

        # print(f"Lower left: {a_region_ll}")

        # check each region of field of view
        for i in np.ndindex((self._fov_w, self._fov_h)):
            if d[0] != 0:  # going left or right
                dx = i[0] - ego_x
                dy = i[1] - ego_y
            else:  # going up or down
                dx = i[1] - ego_x
                dy = i[0] - ego_y

            x, y = (
                a_region_ll[0] + (dx * car_l * x_inc),
                a_region_ll[1] + (dy * car_l * y_inc),
            )
            x_next, y_next = (x + car_l * x_inc, y + car_l * y_inc)

            # print(f"{i} -> ({x}, {y}) : ({x_next}, {y_next})")

            # TODO make argument if grass should be detected
            # check if region is in the grass area
            # if self._is_region_on_road((x, y), (x_next, y_next), scale):
            #     fov[i] = np.array([0, 0, 1, 1, 1, 1])
            #     continue

            for a in self._agents:
                if not a.state.on_the_road or a.state.done:
                    continue

                # default values are for "up" - no rotation
                ax, ay = int(a.state.position[0] * scale), int(a.state.position[1] * scale)
                px, py = p[0], p[1]

                # translate coordinate to center of car
                ad = a.state.direction
                if ad == self._directions["down"]:
                    ay += car_l / 2
                elif ad == self._directions["up"]:
                    ay += -car_l / 2
                elif ad == self._directions["left"]:
                    ax += car_l / 2
                elif ad == self._directions["right"]:
                    ax += -car_l / 2

                if inside_fn((x, y), (x_next, y_next), (ax, ay)):

                    # print(f"Agent {a.index} at {i}.")
                    # print(f"\tPosition: ({ax}, {ay})")
                    # rotate coordinates in order to calculate relative coordinates
                    if d == self._directions["left"]:  # 90 degrees
                        ax, ay = ay, -ax
                        px, py = p[1], -p[0]
                    elif d == self._directions["down"]:  # 180 degrees
                        ax, ay = -ax, -ay
                        px, py = -p[0], -p[1]
                    elif d == self._directions["right"]:  # 270 degrees
                        ax, ay = -ay, ax
                        px, py = -p[1], p[0]

                    relative_pos = (ax - px, ay - py)

                    # polar coordinate
                    r = math.sqrt(relative_pos[0] ** 2 + relative_pos[1] ** 2) / scale
                    theta = math.atan2(relative_pos[1], relative_pos[0])
                    # print(f"\tPolar: ({r}, {theta})")

                    # direction
                    direction = self.__get_direction_one_hot(a)

                    fov[i] = np.concatenate(([r], [theta], direction))

                    others_in_vision[a.index] = 1

        return fov, others_in_vision

    def _get_global_positions(self, agent):
        obs = np.full((self.n_agents, 7), 0.0, dtype=np.float32)
        x = agent.state.position[0]
        y = agent.state.position[1]

        center_x = 0.5
        center_y = 0.5

        for i, a in enumerate(self._agents):
            if not a.state.on_the_road:
                continue

            direction = self.__get_direction_one_hot(a)
            ax = a.state.position[0]
            ay = a.state.position[1]

            if a.state.position is not agent.state.position:
                self_to_center = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                other_to_center = math.sqrt((ax - center_x) ** 2 + (ay - center_y) ** 2)

                manhatten_distance = self_to_center + other_to_center
            else:
                manhatten_distance = 0

            obs[i] = np.concatenate(([ax], [ay], [manhatten_distance], direction))

        return obs

    def get_agent_obs(self):
        obs_dim = spaces.flatdim(self.observation_space[0])
        agent_obs = np.empty((self.n_agents, obs_dim), dtype=np.float32)

        if self.observability == "fov":
            agent_fovs = np.empty((self.n_agents, obs_dim), dtype=np.float32)
            whos = np.empty((self.n_agents, self.n_agents))
            for i, agent in enumerate(self._agents):
                # flatten field of view
                fov, who = self._get_fov(agent)

                agent_fovs[i] = fov.flatten()
                whos[i] = who

            return agent_fovs, whos

        elif self.observability == "global":
            for i, agent in enumerate(self._agents):
                agent_obs[i] = self._get_global_positions(agent).flatten()

        return agent_obs

    def _free_gates(self):
        free_gates = {}

        agent_positions = [agent.state.position for agent in self._agents if agent.state.on_the_road]

        for gate in self._entry_gates:
            free_gates[gate] = (True, self._entry_gates[gate])  # (free; pos)
            gate_pos = self._entry_gates[gate]
            if gate == "top":
                for pos in agent_positions:
                    if pos[1] > gate_pos[1] - 2 * CAR_LENGTH:
                        free_gates[gate] = (False, None)
                        break
            elif gate == "left":
                for pos in agent_positions:
                    if pos[0] < gate_pos[0] + 2 * CAR_LENGTH:
                        free_gates[gate] = (False, None)
                        break
            elif gate == "bottom":
                for pos in agent_positions:
                    if pos[1] < gate_pos[1] + 2 * CAR_LENGTH:
                        free_gates[gate] = (False, None)
                        break
            elif gate == "right":
                for pos in agent_positions:
                    if pos[0] > gate_pos[0] - 2 * CAR_LENGTH:
                        free_gates[gate] = (False, None)
                        break

        return [gate[1] for gate in free_gates.values() if gate[0] == True]

    def step(self, actions):
        self._step_count += 1  # global env step
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        took_step = np.zeros(self.n_agents, dtype=np.float32)  # holds who took a step in this env step
        collisions = np.zeros(self.n_agents, dtype=np.float32)
        unique_collisions = 0  # counts number of new collisions in this step

        for agent_i, action in enumerate(actions):
            agent = self._agents[agent_i]
            if not agent.state.done and agent.state.on_the_road:
                agent.state.step_count += 1
                took_step[agent_i] = 1

                collision_flag, who = self._update_agent_pos(agent, action)
                if collision_flag:
                    collisions[agent_i] = 1
                    unique_collisions += 1
                    agent.state.num_repeated_collisions += 1

                    # print(f"Collision! Reward: {self.collision_cost}")
                    rewards[agent_i] += self.collision_cost

                    # remove agent from episode if it has collided
                    # agent.done()

                    # TODO give the other car a little punishment
                    # rewards[who] += -1

                    agent.state.colliding = (True, who)
                else:
                    agent.state.colliding = (False, None)
                    agent.state.num_repeated_collisions = 0

                rewards[agent_i] += self.step_cost * (1 - actions[agent_i]) * agent.state.step_count
                # rewards[agent_i] += self.step_cost * agent.state.step_count

                # check if agent has reached it's destination
                if not agent.state.done and self._reached_destination(agent):
                    agent.done()
                    self.curr_cars_count -= 1

                # set agent as done if max steps is reached (for this given car by its local step_count)
                if agent.state.step_count >= self.max_steps:
                    agent.done()
                    self.curr_cars_count -= 1

        agent_dones = [agent.state.done for agent in self._agents]

        # add new cars according to probalility _arrive_prob
        if self.np_random.uniform(0, 1) < self.arrive_prob and not all(
            [agent.state.on_the_road for agent in self._agents]
        ):
            free_gates = self._free_gates()
            agents_off_road = [agent for agent in self._agents if not agent.state.on_the_road]

            if len(agents_off_road) > 0 and len(free_gates) > 0:
                agent_to_enter = self.np_random.choice(agents_off_road)
                free_gate_indx = self.np_random.choice(len(free_gates))
                pos = tuple(free_gates[free_gate_indx])

                agent_to_enter.state.position = pos
                agent_to_enter.state.direction = self._route_vectors[pos]
                agent_to_enter.state.on_the_road = True
                agent_to_enter.state.route = self.np_random.randint(1, self._n_routes + 1)  # [1,n_routes] (inclusive)
                self.curr_cars_count += 1

        # update env info
        self._acc_unique_collisions += unique_collisions
        self._total_reward += sum(rewards)
        self._avg_speed = self._avg_speed + (np.array(actions) - self._avg_speed).mean() / self._step_count

        cars_on_road = [a.state.on_the_road for a in self._agents]

        return (
            self.get_agent_obs(),
            self.reward_callback(rewards, self.n_agents),
            agent_dones,
            {
                "collisions": collisions,
                "unique_collisions": unique_collisions,
                "cars_on_road": cars_on_road,
                "took_step": took_step,
            },
        )

    def _check_collision(self, agent, next_pos):
        agent_copy = copy.deepcopy(agent)
        agent_copy.state.position = next_pos

        others = [x for x in self._agents if x.index != agent_copy.index and not x.state.done and x.state.on_the_road]
        for other in others:
            at, ab = agent_copy.get_rect_coords()
            bt, bb = other.get_rect_coords()
            collide = not (at.x < bb.x or ab.x > bt.x or at.y < bb.y or ab.y > bt.y)
            if collide:
                return True, other.index

        return False, None

    def _update_agent_pos(self, agent, move):
        move = move * self.movement_scale_factor  # movement is between [0;1]
        curr_pos = agent.state.position
        direction = agent.state.direction

        # calculate new position by taking the movement in the current direction
        next_pos = (curr_pos[0] + move * direction[0], curr_pos[1] + move * direction[1])

        # if there is a collision
        collision, who = self._check_collision(agent, next_pos)
        if collision:
            return collision, who

        agent.state.position = next_pos  # set agent position
        return False, None

    def _reached_destination(self, agent):
        def has_reached_dest():
            if agent.state.direction == self._directions["down"]:
                return pos[1] < dest[1]
            elif agent.state.direction == self._directions["up"]:
                return pos[1] > dest[1]
            elif agent.state.direction == self._directions["left"]:
                return pos[0] < dest[0]
            elif agent.state.direction == self._directions["right"]:
                return pos[0] > dest[0]

        pos = agent.state.position
        dest = self._destinations[agent.state.direction]
        reached_dest = has_reached_dest()

        return reached_dest

    def _reset_environment(self):
        shuffled_gates = list(self._route_vectors.keys())
        self.np_random.shuffle(shuffled_gates)

        self._agents = [Agent(i) for i in range(self.n_agents)]
        for agent in self._agents:
            agent.state.position = (-1, -1)  # not yet on road

    def render(self, mode="human"):
        if mode != "human":
            super(TrafficJunctionContinuousEnv, self).render(mode=mode)

        import tjc_gym.envs.rendering as rendering

        if self._viewer == None:
            self._viewer = rendering.Viewer(1000, 1000)
            self._viewer.set_bounds(0, ENV_WIDTH, 0, ENV_WIDTH)

        if self._agent_geoms == None:
            self._initialize_env_geoms()

        # update info box
        self._reward_label.text = "{:.2f}".format(self._total_reward)
        self._avg_speed_label.text = "{:.2f}".format(self._avg_speed)
        self._collisions_label.text = str(self._acc_unique_collisions)
        self._steps_label.text = str(self._step_count)

        for i, agent in enumerate(self._agents):
            if not agent.state.done and agent.state.on_the_road:
                new_geom_pos = agent.state.position
                label_pos = agent.state.position
                d = agent.state.direction

                # translate geom pos
                translate = None
                translate_label = None
                if d == self._directions["down"]:
                    translate = (-CAR_WIDTH / 2, 0)
                    translate_label = (0, CAR_LENGTH / 2)
                elif d == self._directions["left"]:
                    translate = (0, CAR_WIDTH / 2)
                    translate_label = (CAR_LENGTH / 2, 0)
                elif d == self._directions["up"]:
                    translate = (-CAR_WIDTH / 2, -CAR_LENGTH)
                    translate_label = (0, -CAR_LENGTH / 2)
                elif d == self._directions["right"]:
                    translate = (-CAR_LENGTH, CAR_WIDTH / 2)
                    translate_label = (-CAR_LENGTH / 2, 0)
                new_geom_pos = tuple(sum(x) for x in zip(new_geom_pos, translate))
                label_pos = tuple(sum(x) for x in zip(label_pos, translate_label))

                # set geom and label to agent position
                self._agent_geoms[i].position = new_geom_pos
                self._agent_labels[i].position = label_pos

                # rotate geom following horizontal route
                if d == self._directions["left"] or d == self._directions["right"]:
                    self._agent_geoms[i].rotation = 90
                    self._agent_labels[i].rotation = 90

            else:
                # outside view
                self._agent_geoms[i].position = (-1, -1)
                self._agent_labels[i].position = (-1, -1)

        self._viewer.render()

    def __get_direction_one_hot(self, agent):
        direction = np.zeros(self._n_directions, dtype=int)
        if not agent.state.direction == None:
            if agent.state.direction == self._directions["down"]:
                direction[0] = 1
            elif agent.state.direction == self._directions["left"]:
                direction[1] = 1
            elif agent.state.direction == self._directions["up"]:
                direction[2] = 1
            elif agent.state.direction == self._directions["right"]:
                direction[3] = 1
        return direction

    def _reset_render(self):
        self._agent_geoms = None
        self._agent_labels = None

        # info box
        self._reward_label = None
        self._avg_speed_label = None
        self._collisions_label = None
        self._steps_label = None

        if not self._viewer == None:
            self._viewer.reset()

    def _initialize_env_geoms(self):
        # cars and labels
        self._agent_geoms = []
        self._agent_labels = []
        for agent in self._agents:
            car = self._viewer.add_rectangle(CAR_WIDTH, CAR_LENGTH)
            car.color = (0, 0, 0)
            self._agent_geoms.append(car)

            label = self._viewer.add_label(str(agent.index))
            self._agent_labels.append(label)

        # grass
        grass_recs = [self._viewer.add_rectangle(GRASS_WIDTH, GRASS_WIDTH, is_foreground=False) for _ in range(4)]
        for rec in grass_recs:
            rec.color = (126, 200, 80)
        grass_recs[0].position = (0, ENV_WIDTH - GRASS_WIDTH)
        grass_recs[1].position = (ENV_WIDTH - GRASS_WIDTH, ENV_WIDTH - GRASS_WIDTH)
        grass_recs[2].position = (0, 0)
        grass_recs[3].position = (ENV_WIDTH - GRASS_WIDTH, 0)

        # road dashes
        num_road_dashes = 10
        dash_length = GRASS_WIDTH / (num_road_dashes * 2 - 1)

        # info box
        x_left = 0.68 * ENV_WIDTH
        self._viewer.add_label("Episode stats", x_left, 0.94 * ENV_WIDTH, size=24, anchor_x="left")

        y_base = 0.88 * ENV_WIDTH
        y_inc = -0.05 * ENV_WIDTH
        x_left_indent = x_left + ENV_WIDTH * 0.18

        self._viewer.add_label("Steps", x_left, y_base, anchor_x="left")
        self._steps_label = self._viewer.add_label(str(self._step_count), x_left_indent, y_base, anchor_x="left")

        self._viewer.add_label("Reward", x_left, y_base + y_inc, anchor_x="left")
        self._reward_label = self._viewer.add_label(
            str(self._total_reward), x_left_indent, y_base + y_inc, anchor_x="left"
        )

        self._viewer.add_label("Avg. speed", x_left, y_base + 2 * y_inc, anchor_x="left")
        self._avg_speed_label = self._viewer.add_label(
            str(self._avg_speed), x_left_indent, y_base + 2 * y_inc, anchor_x="left"
        )

        self._viewer.add_label("Collisions", x_left, y_base + 3 * y_inc, anchor_x="left")
        self._collisions_label = self._viewer.add_label(
            str(self._acc_unique_collisions), x_left_indent, y_base + 3 * y_inc, anchor_x="left"
        )

        def add_dash(start, end):
            line = self._viewer.add_line(start, end, width=0.0025, is_foreground=False)
            line.color = (192, 192, 192)

        for i in range(num_road_dashes * 2 - 1):
            if i % 2 == 1:
                continue
            # right
            start = (ENV_WIDTH - GRASS_WIDTH + i * dash_length, ENV_WIDTH / 2)
            end = (start[0] + dash_length, ENV_WIDTH / 2)
            add_dash(start, end)

            # left
            start = (0 + i * dash_length, ENV_WIDTH / 2)
            end = (start[0] + dash_length, ENV_WIDTH / 2)
            add_dash(start, end)

            # up
            start = (ENV_WIDTH / 2, ENV_WIDTH - GRASS_WIDTH + i * dash_length)
            end = (ENV_WIDTH / 2, start[1] + dash_length)
            add_dash(start, end)

            # down
            start = (ENV_WIDTH / 2, GRASS_WIDTH - i * dash_length)
            end = (ENV_WIDTH / 2, start[1] - dash_length)
            add_dash(start, end)
