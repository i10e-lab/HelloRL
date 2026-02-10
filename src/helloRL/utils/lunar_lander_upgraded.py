__credits__ = ["Andrea PIERRÉ"]

import math
from typing import TYPE_CHECKING

import numpy as np

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import EzPickle
from gymnasium.utils.step_api_compatibility import step_api_compatibility


try:
    import Box2D
    from Box2D.b2 import (
        circleShape,
        contactListener,
        edgeShape,
        fixtureDef,
        polygonShape,
        revoluteJointDef,
    )
except ImportError as e:
    raise DependencyNotInstalled(
        'Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`'
    ) from e


if TYPE_CHECKING:
    import pygame


FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder

LANDER_POLY = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14
SIDE_ENGINE_AWAY = 12
MAIN_ENGINE_Y_LOCATION = (
    4  # The Y location of the main engine on the body of the Lander.
)

VIEWPORT_W = 600
VIEWPORT_H = 400

# Lander and leg border customization
LANDER_BORDER_WIDTH = 2  # Border thickness in pixels (0 = no border)
LANDER_BORDER_COLOR = (187, 172, 250)  # Border color (R, G, B) - white by default
LEG_BORDER_WIDTH = 2  # Leg border thickness in pixels (0 = no border)
LEG_BORDER_COLOR = (187, 172, 250)  # Leg border color (R, G, B) - white by default
LANDER_CORNER_RADIUS = 0  # Corner radius for rounded lander edges (0 = sharp corners)


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if (
            self.env.lander == contact.fixtureA.body
            or self.env.lander == contact.fixtureB.body
        ):
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


class LunarLander(gym.Env, EzPickle):
    r"""
    ## Description
    This environment is a classic rocket trajectory optimization problem.
    According to Pontryagin's maximum principle, it is optimal to fire the
    engine at full throttle or turn it off. This is the reason why this
    environment has discrete actions: engine on or off.

    There are two environment versions: discrete or continuous.
    The landing pad is always at coordinates (0,0). The coordinates are the
    first two numbers in the state vector.
    Landing outside of the landing pad is possible. Fuel is infinite, so an agent
    can learn to fly and then land on its first attempt.

    To see a heuristic landing, run:
    ```shell
    python gymnasium/envs/box2d/lunar_lander.py
    ```

    ## Action Space
    There are four discrete actions available:
    - 0: do nothing
    - 1: fire left orientation engine
    - 2: fire main engine
    - 3: fire right orientation engine

    ## Observation Space
    The state is an 8-dimensional vector: the coordinates of the lander in `x` & `y`, its linear
    velocities in `x` & `y`, its angle, its angular velocity, and two booleans
    that represent whether each leg is in contact with the ground or not.

    ## Rewards
    After every step a reward is granted. The total reward of an episode is the
    sum of the rewards for all the steps within that episode.

    For each step, the reward:
    - is increased/decreased the closer/further the lander is to the landing pad.
    - is increased/decreased the slower/faster the lander is moving.
    - is decreased the more the lander is tilted (angle not horizontal).
    - is increased by 10 points for each leg that is in contact with the ground.
    - is decreased by 0.03 points each frame a side engine is firing.
    - is decreased by 0.3 points each frame the main engine is firing.

    The episode receive an additional reward of -100 or +100 points for crashing or landing safely respectively.

    An episode is considered a solution if it scores at least 200 points.

    ## Starting State
    The lander starts at the top center of the viewport with a random initial
    force applied to its center of mass.

    ## Episode Termination
    The episode finishes if:
    1) the lander crashes (the lander body gets in contact with the moon);
    2) the lander gets outside of the viewport (`x` coordinate is greater than 1);
    3) the lander is not awake. From the [Box2D docs](https://box2d.org/documentation/md__d_1__git_hub_box2d_docs_dynamics.html#autotoc_md61),
        a body which is not awake is a body which doesn't move and doesn't
        collide with any other body:
    > When Box2D determines that a body (or group of bodies) has come to rest,
    > the body enters a sleep state which has very little CPU overhead. If a
    > body is awake and collides with a sleeping body, then the sleeping body
    > wakes up. Bodies will also wake up if a joint or contact attached to
    > them is destroyed.

    ## Arguments

    Lunar Lander has a large number of arguments

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
    ...                enable_wind=False, wind_power=15.0, turbulence_power=1.5)
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<LunarLander<LunarLander-v3>>>>>

    ```

     * `continuous` determines if discrete or continuous actions (corresponding to the throttle of the engines) will be used with the
     action space being `Discrete(4)` or `Box(-1, +1, (2,), dtype=np.float32)` respectively.
     For continuous actions, the first coordinate of an action determines the throttle of the main engine, while the second
     coordinate specifies the throttle of the lateral boosters. Given an action `np.array([main, lateral])`, the main
     engine will be turned off completely if `main < 0` and the throttle scales affinely from 50% to 100% for
     `0 <= main <= 1` (in particular, the main engine doesn't work  with less than 50% power).
     Similarly, if `-0.5 < lateral < 0.5`, the lateral boosters will not fire at all. If `lateral < -0.5`, the left
     booster will fire, and if `lateral > 0.5`, the right booster will fire. Again, the throttle scales affinely
     from 50% to 100% between -1 and -0.5 (and 0.5 and 1, respectively).

    * `gravity` dictates the gravitational constant, this is bounded to be within 0 and -12. Default is -10.0

    * `enable_wind` determines if there will be wind effects applied to the lander. The wind is generated using
     the function `tanh(sin(2 k (t+C)) + sin(pi k (t+C)))` where `k` is set to 0.01 and `C` is sampled randomly between -9999 and 9999.

    * `wind_power` dictates the maximum magnitude of linear wind applied to the craft. The recommended value for
     `wind_power` is between 0.0 and 20.0.

    * `turbulence_power` dictates the maximum magnitude of rotational wind applied to the craft.
     The recommended value for `turbulence_power` is between 0.0 and 2.0.

    ## Version History
    - v3:
        - Reset wind and turbulence offset (`C`) whenever the environment is reset to ensure statistical independence between consecutive episodes (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/954)).
        - Fix non-deterministic behaviour due to not fully destroying the world (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/728)).
        - Changed observation space for `x`, `y`  coordinates from $\pm 1.5$ to $\pm 2.5$, velocities from $\pm 5$ to $\pm 10$ and angles from $\pm \pi$ to $\pm 2\pi$ (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/752)).
    - v2: Count energy spent and in v0.24, added turbulence with wind power and turbulence_power parameters
    - v1: Legs contact with ground added in state vector; contact with ground give +10 reward points, and -10 if then lose contact; reward renormalized to 200; harder initial random push.
    - v0: Initial version

    ## Notes

    There are several unexpected bugs with the implementation of the environment.

    1. The position of the side thrusters on the body of the lander changes, depending on the orientation of the lander.
    This in turn results in an orientation dependent torque being applied to the lander.

    2. The units of the state are not consistent. I.e.
    * The angular velocity is in units of 0.4 radians per second. In order to convert to radians per second, the value needs to be multiplied by a factor of 2.5.

    For the default values of VIEWPORT_W, VIEWPORT_H, SCALE, and FPS, the scale factors equal:
    'x': 10, 'y': 6.666, 'vx': 5, 'vy': 7.5, 'angle': 1, 'angular velocity': 2.5

    After the correction has been made, the units of the state are as follows:
    'x': (units), 'y': (units), 'vx': (units/second), 'vy': (units/second), 'angle': (radians), 'angular velocity': (radians/second)

    <!-- ## References -->

    ## Credits
    Created by Oleg Klimov
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        continuous: bool = False,
        gravity: float = -10.0,
        enable_wind: bool = False,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
    ):
        EzPickle.__init__(
            self,
            render_mode,
            continuous,
            gravity,
            enable_wind,
            wind_power,
            turbulence_power,
        )

        assert (
            -12.0 < gravity and gravity < 0.0
        ), f"gravity (current value: {gravity}) must be between -12 and 0"
        self.gravity = gravity

        if 0.0 > wind_power or wind_power > 20.0:
            gym.logger.warn(
                f"wind_power value is recommended to be between 0.0 and 20.0, (current value: {wind_power})"
            )
        self.wind_power = wind_power

        if 0.0 > turbulence_power or turbulence_power > 2.0:
            gym.logger.warn(
                f"turbulence_power value is recommended to be between 0.0 and 2.0, (current value: {turbulence_power})"
            )
        self.turbulence_power = turbulence_power

        self.enable_wind = enable_wind

        self.screen: pygame.Surface = None
        self.clock = None
        self.isopen = True
        self.world = Box2D.b2World(gravity=(0, gravity))
        self.moon = None
        self.lander: Box2D.b2Body | None = None
        self.particles = []

        self.prev_reward = None

        # Star field for visual enhancement
        self.stars = None
        self.star_twinkle_offsets = None

        # Lunar surface tile
        self.lunar_tile = None

        self.continuous = continuous

        low = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                -2.5,  # x coordinate
                -2.5,  # y coordinate
                # velocity bounds is 5x rated speed
                -10.0,
                -10.0,
                -2 * math.pi,
                -10.0,
                -0.0,
                -0.0,
            ]
        ).astype(np.float32)
        high = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                2.5,  # x coordinate
                2.5,  # y coordinate
                # velocity bounds is 5x rated speed
                10.0,
                10.0,
                2 * math.pi,
                10.0,
                1.0,
                1.0,
            ]
        ).astype(np.float32)

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(low, high)

        if self.continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        else:
            # Nop, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(4)

        self.render_mode = render_mode

    def _destroy(self):
        if not self.moon:
            return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        self._destroy()

        # Bug's workaround for: https://github.com/Farama-Foundation/Gymnasium/issues/728
        # Not sure why the self._destroy() is not enough to clean(reset) the total world environment elements, need more investigation on the root cause,
        # we must create a totally new world for self.reset(), or the bug#728 will happen
        self.world = Box2D.b2World(gravity=(0, self.gravity))
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        # Create Terrain
        CHUNKS = 11
        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self.helipad_y = H / 4
        height[CHUNKS // 2 - 2] = self.helipad_y
        height[CHUNKS // 2 - 1] = self.helipad_y
        height[CHUNKS // 2 + 0] = self.helipad_y
        height[CHUNKS // 2 + 1] = self.helipad_y
        height[CHUNKS // 2 + 2] = self.helipad_y
        smooth_y = [
            0.33 * (height[i - 1] + height[i + 0] + height[i + 1])
            for i in range(CHUNKS)
        ]

        self.moon = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(0, 0), (W, 0)])
        )
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        # Create Lander body
        initial_y = VIEWPORT_H / SCALE
        initial_x = VIEWPORT_W / SCALE / 2
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]
                ),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0,
            ),  # 0.99 bouncy
        )
        self.lander.color1 = (128, 102, 230)
        self.lander.color2 = (77, 77, 128)

        # Apply the initial random impulse to the lander
        self.lander.ApplyForceToCenter(
            (
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            ),
            True,
        )

        if self.enable_wind:  # Initialize wind pattern based on index
            self.wind_idx = self.np_random.integers(-9999, 9999)
            self.torque_idx = self.np_random.integers(-9999, 9999)

        # Create Lander Legs
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001,
                ),
            )
            leg.ground_contact = False
            leg.color1 = (128, 102, 230)
            leg.color2 = (77, 77, 128)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i,  # low enough not to jump back into the sky
            )
            if i == -1:
                rjd.lowerAngle = (
                    +0.9 - 0.5
                )  # The most esoteric numbers here, angled legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs

        if self.render_mode == "human":
            self.render()
        return self.step(np.array([0, 0]) if self.continuous else 0)[0], {}

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=2 / SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3,
            ),
        )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all_particle):
        while self.particles and (all_particle or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def _create_stars(self):
        """Create a starfield with various star sizes and brightness for twinkling effect."""
        if self.stars is None:
            num_stars = 100
            self.stars = []
            for _ in range(num_stars):
                x = self.np_random.integers(0, VIEWPORT_W)
                y = self.np_random.integers(0, VIEWPORT_H)
                size = self.np_random.choice([1, 1, 1, 1, 1, 2])  # mostly 1-pixel stars, rarely 2
                brightness = self.np_random.uniform(0.4, 0.8)  # bright stars, visible on dark sky only
                twinkle_speed = self.np_random.uniform(0.02, 0.08)
                self.stars.append({
                    'x': x,
                    'y': y,
                    'size': size,
                    'base_brightness': brightness,
                    'twinkle_speed': twinkle_speed
                })
            self.star_twinkle_offsets = [
                self.np_random.uniform(0, 2 * math.pi) for _ in range(num_stars)
            ]

    def _load_lunar_tile(self):
        """Load and create tiled lunar surface."""
        if self.lunar_tile is None:
            try:
                import pygame
                import os

                # Try to load the tile image
                tile_path = os.path.join(os.path.dirname(__file__), 'assets/tile.png')
                if os.path.exists(tile_path):
                    tile = pygame.image.load(tile_path)

                    # Scale down the tile to make it smaller (0.25x size)
                    tile_w, tile_h = tile.get_size()
                    scaled_tile = pygame.transform.scale(
                        tile,
                        (int(tile_w * 0.5), int(tile_h * 0.5))
                    )
                    # Flip the tile vertically to correct orientation
                    scaled_tile = pygame.transform.flip(scaled_tile, False, True)
                    tile_w, tile_h = scaled_tile.get_size()

                    # Create a surface large enough to tile across the viewport
                    tiled_surface = pygame.Surface((VIEWPORT_W, VIEWPORT_H))

                    # Tile the image across the surface
                    for x in range(0, VIEWPORT_W, tile_w):
                        for y in range(0, VIEWPORT_H, tile_h):
                            tiled_surface.blit(scaled_tile, (x, y))

                    self.lunar_tile = tiled_surface
                else:
                    # Fallback to solid color if tile not found
                    self.lunar_tile = None
            except Exception:
                # Fallback to solid color if loading fails
                self.lunar_tile = None

    def _point_in_polygon(self, x, y, poly):
        """Check if point (x, y) is inside polygon using ray casting algorithm."""
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(1, n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _get_rounded_polygon(self, vertices, radius, num_segments=20):
        """Generate a polygon with rounded corners using quadratic bezier curves.
        
        Args:
            vertices: List of (x, y) tuples representing polygon vertices
            radius: Corner radius in pixels
            num_segments: Number of segments to use for each rounded corner
            
        Returns:
            List of (x, y) tuples representing the rounded polygon
        """
        if radius <= 0 or len(vertices) < 3:
            return vertices
        
        rounded = []
        n = len(vertices)
        
        for i in range(n):
            # Get current vertex and adjacent vertices
            p0 = vertices[(i - 1) % n]
            p1 = vertices[i]
            p2 = vertices[(i + 1) % n]
            
            # Calculate vectors from current vertex to adjacent vertices
            v1 = (p0[0] - p1[0], p0[1] - p1[1])
            v2 = (p2[0] - p1[0], p2[1] - p1[1])
            
            # Calculate lengths
            len1 = math.sqrt(v1[0]**2 + v1[1]**2)
            len2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if len1 == 0 or len2 == 0:
                rounded.append(p1)
                continue
            
            # Normalize vectors
            v1_norm = (v1[0] / len1, v1[1] / len1)
            v2_norm = (v2[0] / len2, v2[1] / len2)
            
            # Limit radius to half the shortest edge
            max_radius = min(len1, len2) / 2
            r = min(radius, max_radius)
            
            # Calculate start and end points of the curve
            start = (p1[0] + v1_norm[0] * r, p1[1] + v1_norm[1] * r)
            end = (p1[0] + v2_norm[0] * r, p1[1] + v2_norm[1] * r)
            
            # Use quadratic bezier curve with p1 as control point
            for j in range(num_segments + 1):
                t = j / num_segments
                # Quadratic bezier: B(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
                x = (1-t)**2 * start[0] + 2*(1-t)*t * p1[0] + t**2 * end[0]
                y = (1-t)**2 * start[1] + 2*(1-t)*t * p1[1] + t**2 * end[1]
                rounded.append((int(round(x)), int(round(y))))
        
        return rounded

    def step(self, action):
        assert self.lander is not None

        # Update wind and apply to the lander
        assert self.lander is not None, "You forgot to call reset()"
        if self.enable_wind and not (
            self.legs[0].ground_contact or self.legs[1].ground_contact
        ):
            # the function used for wind is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            wind_mag = (
                math.tanh(
                    math.sin(0.02 * self.wind_idx)
                    + (math.sin(math.pi * 0.01 * self.wind_idx))
                )
                * self.wind_power
            )
            self.wind_idx += 1
            self.lander.ApplyForceToCenter(
                (wind_mag, 0.0),
                True,
            )

            # the function used for torque is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            torque_mag = (
                math.tanh(
                    math.sin(0.02 * self.torque_idx)
                    + (math.sin(math.pi * 0.01 * self.torque_idx))
                )
                * self.turbulence_power
            )
            self.torque_idx += 1
            self.lander.ApplyTorque(
                torque_mag,
                True,
            )

        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float64)
        else:
            assert self.action_space.contains(
                action
            ), f"{action!r} ({type(action)}) invalid "

        # Apply Engine Impulses

        # Tip is the (X and Y) components of the rotation of the lander.
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))

        # Side is the (-Y and X) components of the rotation of the lander.
        side = (-tip[1], tip[0])

        # Generate two random numbers between -1/SCALE and 1/SCALE.
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (
            not self.continuous and action == 2
        ):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0

            # 4 is move a bit downwards, +-2 for randomness
            # The components of the impulse to be applied by the main engine.
            ox = (
                tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                + side[0] * dispersion[1]
            )
            oy = (
                -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                - side[1] * dispersion[1]
            )

            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            if self.render_mode is not None:
                # particles are just a decoration, with no impact on the physics, so don't add them when not rendering
                p = self._create_particle(
                    3.5,  # 3.5 is here to make particle speed adequate
                    impulse_pos[0],
                    impulse_pos[1],
                    m_power,
                )
                p.ApplyLinearImpulse(
                    (
                        ox * MAIN_ENGINE_POWER * m_power,
                        oy * MAIN_ENGINE_POWER * m_power,
                    ),
                    impulse_pos,
                    True,
                )
            self.lander.ApplyLinearImpulse(
                (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (
            not self.continuous and action in [1, 3]
        ):
            # Orientation/Side engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                # action = 1 is left, action = 3 is right
                direction = action - 2
                s_power = 1.0

            # The components of the impulse to be applied by the side engines.
            ox = tip[0] * dispersion[0] + side[0] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            oy = -tip[1] * dispersion[0] - side[1] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )

            # The constant 17 is a constant, that is presumably meant to be SIDE_ENGINE_HEIGHT.
            # However, SIDE_ENGINE_HEIGHT is defined as 14
            # This causes the position of the thrust on the body of the lander to change, depending on the orientation of the lander.
            # This in turn results in an orientation dependent torque being applied to the lander.
            impulse_pos = (
                self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )
            if self.render_mode is not None:
                # particles are just a decoration, with no impact on the physics, so don't add them when not rendering
                p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
                p.ApplyLinearImpulse(
                    (
                        ox * SIDE_ENGINE_POWER * s_power,
                        oy * SIDE_ENGINE_POWER * s_power,
                    ),
                    impulse_pos,
                    True,
                )
            self.lander.ApplyLinearImpulse(
                (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity

        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]
        assert len(state) == 8

        reward = 0
        shaping = (
            -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100 * abs(state[4])
            + 10 * state[6]
            + 10 * state[7]
        )  # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= (
            m_power * 0.30
        )  # less fuel spent is better, about -30 for heuristic landing
        reward -= s_power * 0.03

        terminated = False
        if self.game_over or abs(state[0]) >= 1.0:
            terminated = True
            reward = -100
        if not self.lander.awake:
            terminated = True
            reward = +100

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(state, dtype=np.float32), reward, terminated, False, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[box2d]"`'
            ) from e

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        
        # Ensure display is initialized for convert_alpha even if not in human mode (e.g. rgb_array)
        if not pygame.display.get_init():
             pygame.display.init()
             # We need a display mode for convert() and convert_alpha() to work, 
             # even a hidden one or the smallest one if 'screen' isn't set.
             # Note: In rgb_array mode, we might not want a visible window, but pygame still often needs one for pixel conversions.
             # However, simple surfaces work without display.set_mode, EXCEPT convert_alpha might check for a display context.
             pass

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))

        pygame.transform.scale(self.surf, (SCALE, SCALE))

        # Load and draw tiled lunar surface background
        self._load_lunar_tile()
        if self.lunar_tile is not None:
            # Use tiled texture for background
            self.surf.blit(self.lunar_tile, (0, 0))
        else:
            # Fallback to solid color
            self.surf.fill((220, 220, 225))

        # Load background image if not already loaded
        if not hasattr(self, 'bg_img'):
            import os
            self.bg_img = None
            bg_filename = "assets/background.png"
            possible_paths = [
                os.path.join(os.path.dirname(__file__), bg_filename),
                os.path.join(os.getcwd(), bg_filename),
                bg_filename
            ]
            found_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    found_path = p
                    break
            
            if found_path:
                try:
                    # If display is initialized, convert_alpha optimizes format. 
                    # If not, we just load.
                    loaded_img = pygame.image.load(found_path)
                    if pygame.display.get_init() and pygame.display.get_surface() is not None:
                        loaded_img = loaded_img.convert_alpha()
                    
                    self.bg_img = pygame.transform.scale(loaded_img, (VIEWPORT_W, VIEWPORT_H))
                    # Flip image vertically because the coordinate system might be inverted
                    self.bg_img = pygame.transform.flip(self.bg_img, False, True)
                except Exception as e:
                    pass

        # Draw sky (either image or solid color)
        if self.bg_img:
            # Create a mask for the sky
            sky_mask = pygame.Surface((VIEWPORT_W, VIEWPORT_H), pygame.SRCALPHA)
            sky_mask.fill((0, 0, 0, 0))
            
            for p in self.sky_polys:
                scaled_poly = []
                for coord in p:
                    scaled_poly.append((int(coord[0] * SCALE), int(coord[1] * SCALE)))
                # Draw white polygons on mask where we want the sky image to appear
                # Using pygame.draw.polygon is more robust for masks than gfxdraw
                pygame.draw.polygon(sky_mask, (255, 255, 255, 255), scaled_poly)
            
            # Create a copy of background to mask
            bg_layer = self.bg_img.copy()
            # Multiply: Image * Mask. 
            # Where Mask is White(1,1,1,1) -> Image stays. 
            # Where Mask is Transparent(0,0,0,0) -> Transparent.
            bg_layer.blit(sky_mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            
            # Blit the masked background onto the main surface
            self.surf.blit(bg_layer, (0, 0))
            
        else:
            # Draw dark blue sky polygons (above terrain)
            for p in self.sky_polys:
                # p = [p1, p2, (p2[0], H), (p1[0], H)]
                # Draw sky from terrain UP to top of viewport
                scaled_poly = []
                for coord in p:
                    scaled_poly.append((int(coord[0] * SCALE), int(coord[1] * SCALE)))
                # Dark blue sky color
                sky_color = (8, 10, 20)
                gfxdraw.filled_polygon(self.surf, scaled_poly, sky_color)
                gfxdraw.aapolygon(self.surf, scaled_poly, sky_color)

        # Draw black outline on terrain surface (boundary between sky and ground)
        for p in self.sky_polys:
             # p = [p1, p2, (p2[0], H), (p1[0], H)]
             # The ground segment is from p1 to p2
             p1 = (int(p[0][0] * SCALE), int(p[0][1] * SCALE))
             p2 = (int(p[1][0] * SCALE), int(p[1][1] * SCALE))
             # Draw anti-aliased black line
             pygame.draw.aaline(self.surf, (167, 167, 167), p1, p2)
             # Draw it again slightly offset or create a polygon for thickness if needed, 
             # but a single aaline is usually cleanest for "outline".

        # Draw twinkling stars on the sky (only in sky area)
        self._create_stars()
        if self.stars is not None:
            for i, star in enumerate(self.stars):
                # Check if star is within any sky polygon
                is_in_sky = False
                for p in self.sky_polys:
                    # Convert polygon to screen coordinates
                    scaled_poly = [(coord[0] * SCALE, coord[1] * SCALE) for coord in p]
                    if self._point_in_polygon(star['x'], star['y'], scaled_poly):
                        is_in_sky = True
                        break

                # Only draw star if it's in the sky area
                if is_in_sky:
                    # Calculate twinkle effect using sine wave
                    twinkle_phase = self.star_twinkle_offsets[i]
                    self.star_twinkle_offsets[i] += star['twinkle_speed']
                    twinkle_factor = 0.7 + 0.3 * math.sin(twinkle_phase)
                    brightness = star['base_brightness'] * twinkle_factor

                    # Star color with slight blue/white variation
                    color_value = int(brightness * 255)
                    star_color = (color_value, color_value, min(255, int(color_value * 1.1)))

                    # Draw star with anti-aliasing for larger stars
                    if star['size'] == 1:
                        self.surf.set_at((star['x'], star['y']), star_color)
                    else:
                        gfxdraw.filled_circle(self.surf, star['x'], star['y'], star['size'], star_color)
                        gfxdraw.aacircle(self.surf, star['x'], star['y'], star['size'], star_color)
        
        for obj in self.particles:

            r_range = (226, 226)
            g_range = (85, 177)
            b_range = (77, 92)

            progress = (1 - obj.ttl)
            progress = max(0, min(1.0, progress))# ** 2
            color_progress = progress ** 1.0
            fade_progress = progress ** 1.0

            r = int(r_range[0] + (r_range[1] - r_range[0]) * color_progress)
            g = int(g_range[0] + (g_range[1] - g_range[0]) * color_progress)
            b = int(b_range[0] + (b_range[1] - b_range[0]) * color_progress)
            obj.color1 = (
                r, g, b, max(0, int((1 - fade_progress) * 255))
            )
            # obj.color1 = (
            #     int(max(0.2, 0.15 + obj.ttl) * 255),
            #     int(max(0.2, 0.5 * obj.ttl) * 255),
            #     int(max(0.2, 0.5 * obj.ttl) * 255),
            # )
            # obj.color2 = (
            #     int(max(0.2, 0.15 + obj.ttl) * 255),
            #     int(max(0.2, 0.5 * obj.ttl) * 255),
            #     int(max(0.2, 0.5 * obj.ttl) * 255),
            # )
            obj.ttl -= 0.1

        self._clean_particles(False)

        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    center = trans * f.shape.pos * SCALE
                    cx, cy = int(center[0]), int(center[1])
                    radius = int(f.shape.radius * SCALE)
                    # Anti-aliased filled circle
                    gfxdraw.filled_circle(self.surf, cx, cy, radius, obj.color1)
                    gfxdraw.aacircle(self.surf, cx, cy, radius, obj.color1)
                    # gfxdraw.aacircle(self.surf, cx, cy, radius, obj.color2)

                else:
                    path = [(int((trans * v * SCALE)[0]), int((trans * v * SCALE)[1])) for v in f.shape.vertices]
                    
                    # Apply rounded corners to the lander
                    if obj == self.lander and LANDER_CORNER_RADIUS > 0:
                        path = self._get_rounded_polygon(path, LANDER_CORNER_RADIUS)
                    
                    gfxdraw.filled_polygon(self.surf, path, obj.color1)
                    gfxdraw.aapolygon(self.surf, path, obj.color1)
                    # gfxdraw.aapolygon(self.surf, path, obj.color2)

                    # Draw customizable borders for lander and legs (using anti-aliased polygon)
                    if obj == self.lander and LANDER_BORDER_WIDTH > 0:
                        gfxdraw.aapolygon(self.surf, path, LANDER_BORDER_COLOR)
                    elif obj in self.legs and LEG_BORDER_WIDTH > 0:
                        gfxdraw.aapolygon(self.surf, path, LEG_BORDER_COLOR)

        # Draw flags (outside the fixture loop)
        for x in [self.helipad_x1, self.helipad_x2]:
            x = int(x * SCALE)
            flagy1 = int(self.helipad_y * SCALE)
            flagy2 = flagy1 + 50
            
            # Flag pole
            pole_width = 4
            pole_color = (120, 65, 36)
            pole_poly = [
                (x - pole_width // 2, flagy1),
                (x - pole_width // 2, flagy2),
                (x + pole_width // 2, flagy2),
                (x + pole_width // 2, flagy1)
            ]
            gfxdraw.filled_polygon(self.surf, pole_poly, pole_color)
            gfxdraw.aapolygon(self.surf, pole_poly, (0, 0, 0))

            # Flag
            flag_color = (227, 179, 48)
            # Make flag taller: top-left (on pole), bottom-left (on pole), tip (right)
            # flagy2 is top of pole.
            flag_points = [
                (x, flagy2),              # Top attachment point
                (x, flagy2 - 15),         # Bottom attachment point (made taller, was -10)
                (x + 25, flagy2 - 7.5)    # Tip (centered vertically between attach points)
            ]
            gfxdraw.filled_polygon(self.surf, flag_points, flag_color)
            gfxdraw.aapolygon(self.surf, flag_points, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def heuristic(env, s):
    """
    The heuristic for
    1. Testing
    2. Demonstration rollout.

    Args:
        env: The environment
        s (list): The state. Attributes:
            s[0] is the horizontal coordinate
            s[1] is the vertical coordinate
            s[2] is the horizontal speed
            s[3] is the vertical speed
            s[4] is the angle
            s[5] is the angular speed
            s[6] 1 if first leg has contact, else 0
            s[7] 1 if second leg has contact, else 0

    Returns:
         a: The heuristic to be fed into the step function defined above to determine the next step and reward.
    """

    angle_targ = s[0] * 0.5 + s[2] * 1.0  # angle should point towards center
    if angle_targ > 0.4:
        angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4:
        angle_targ = -0.4
    hover_targ = 0.55 * np.abs(
        s[0]
    )  # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
    hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5

    if s[6] or s[7]:  # legs have contact
        angle_todo = 0
        hover_todo = (
            -(s[3]) * 0.5
        )  # override to reduce fall speed, that's all we need after contact

    if env.unwrapped.continuous:
        a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
    return a


def demo_heuristic_lander(env, seed=None, render=False):
    total_reward = 0
    steps = 0
    s, info = env.reset(seed=seed)
    while True:
        a = heuristic(env, s)
        s, r, terminated, truncated, info = step_api_compatibility(env.step(a), True)
        total_reward += r

        if render:
            still_open = env.render()
            if still_open is False:
                break

        if steps % 20 == 0 or terminated or truncated:
            print("observations:", " ".join([f"{x:+0.2f}" for x in s]))
            print(f"step {steps} total_reward {total_reward:+0.2f}")
        steps += 1
        if terminated or truncated:
            break
    if render:
        env.close()
    return total_reward


class LunarLanderContinuous:
    def __init__(self):
        raise error.Error(
            "Error initializing LunarLanderContinuous Environment.\n"
            "Currently, we do not support initializing this mode of environment by calling the class directly.\n"
            "To use this environment, instead create it by specifying the continuous keyword in gym.make, i.e.\n"
            'gym.make("LunarLander-v3", continuous=True)'
        )


if __name__ == "__main__":
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    demo_heuristic_lander(env, render=True)
