# noqa: D212, D415
"""
# GPSD - GPS Denied Coverage Environment

This environment is part of the <a href='..'>MPE environments</a>.

| Import               | `from pettingzoo.mpe.gpsd import gpsd`               |
|----------------------|-------------------------------------------------------|
| Actions              | Discrete                                              |
| Parallel API         | Yes                                                   |
| Manual Control       | No                                                    |
| Agents               | `agents= [agent_0, agent_1, agent_2, agent_3, agent_4]` |
| Agents               | 5                                                     |
| Action Shape         | (5)                                                   |
| Action Values        | Discrete(5)                                           |
| Observation Shape    | (30,)                                                 |
| Observation Values   | (-inf,inf)                                            |


This environment has N_a agents in a 20x20 grid (normalized coordinates [-1, 1]).
A GPS denied zone occupies the center 10x10 square ([-0.5, 0.5]). The zone is
discretized into a grid of cells with width `cell_width`, and a point of interest
is placed at the center of each cell.

Agents move at constant speed with the only control input being the turn rate (omega),
discretized into 5 values linearly spaced between [min_w, max_w]. Physics follow the
unicycle model from core.py's `integrate_state`.

**Coverage**: An agent "covers" a point of interest when:
  1. It is within `r_c` distance of the POI, AND
  2. Its position covariance trace is less than `cov_c`.

Inside the GPS denied zone, the agent's position covariance grows via EKF prediction
(uncertainty accumulates). Outside the zone, GPS is available and covariance resets to zero.
This uses core.py's `update_agent_covariance` and `is_in_gpsd_zone`.

**Reward**: Inversely proportional to position covariance upon coverage (+10 / (1 + cov)).
Agents are penalized for long paths (-0.01 per step) and for being far from uncovered POIs.

Agent observations: `[self_heading, self_pos, self_cov_trace,
                      poi_rel_positions, other_agent_rel_positions,
                      other_agent_comm (range, cov_trace)]`

Agent action space: `[5 discretized turn rates between min_w and max_w]`

### Arguments

``` python
gpsd.env(N_a=5, cell_width=0.25, local_ratio=0.5, max_cycles=100,
         min_w=-1.0, max_w=1.0, speed=1.0, r_c=0.3, cov_c=0.5, p_noise=0.1, r_cov=0.01)
```

`N_a`:  number of agents

`cell_width`:  width of each grid cell in the GPS denied zone (determines number of POIs)

`local_ratio`:  weight between local and global reward

`max_cycles`:  number of frames (a step for each agent) until game terminates

`min_w` / `max_w`:  turn rate bounds (rad/s)

`speed`:  constant forward speed of agents

`r_c`:  communication radius for inter-agent range measurements (also coverage distance threshold)

`cov_c`:  coverage covariance trace threshold

`p_noise`:  process noise for EKF covariance prediction

`r_cov`:  range measurement noise variance (sigma^2 of Gaussian noise added to range)

"""

import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        N_a=5,
        cell_width=0.25,
        local_ratio=0.1,
        max_cycles=100,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
        min_w=-1.0,
        max_w=1.0,
        speed=0.2,
        r_c=0.5,
        cov_c=0.15,
        p_noise=0.1,
        r_cov=0.01,
    ):
        EzPickle.__init__(
            self,
            N_a=N_a,
            cell_width=cell_width,
            local_ratio=local_ratio,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
            dynamic_rescaling=dynamic_rescaling,
            min_w=min_w,
            max_w=max_w,
            speed=speed,
            r_c=r_c,
            cov_c=cov_c,
            p_noise=p_noise,
            r_cov=r_cov,
        )
        assert (
            0.0 <= local_ratio <= 1.0
        ), "local_ratio is a proportion. Must be between 0 and 1."

        # 5 discretized turn rate values (matches action space dim_p*2+1 = 5)
        self.omega_values = np.linspace(min_w, max_w, 5)

        scenario = Scenario(r_c=r_c, cov_c=cov_c)
        world = scenario.make_world(N_a, cell_width=cell_width, speed=speed, p_noise=p_noise, r_c=r_c, r_cov=r_cov)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            local_ratio=local_ratio,
            dynamic_rescaling=dynamic_rescaling,
        )
        self.metadata["name"] = "gpsd_v1"

    def _set_action(self, action, agent, action_space, time=None):
        """Override: map discrete action index to a scalar turn rate (omega).

        The base SimpleEnv maps actions to 2D force vectors, but our unicycle
        model in core.py's integrate_state expects a scalar omega. This override
        produces a numpy scalar that flows through apply_action_omega → integrate_state.
        """
        agent.action.c = np.zeros(self.world.dim_c)

        if agent.movable:
            # Map discrete action (0-4) to scalar omega value
            agent.action.u = np.float64(self.omega_values[action[0]])
            action = action[1:]

        if not agent.silent:
            if self.continuous_actions:
                agent.action.c = action[0]
            else:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            action = action[1:]

        assert len(action) == 0

    def draw(self):
        """Override draw to add GPS denied zone visualization and communication links."""
        import pygame
        
        # Call parent draw to render entities
        super().draw()
        
        # Get current camera range for coordinate transformation
        all_poses = [entity.state.p_pos for entity in self.world.entities]
        cam_range = np.max(np.abs(np.array(all_poses)))
        
        def world_to_screen(pos):
            """Convert world coordinates to screen coordinates."""
            x, y = pos
            y *= -1  # flip y axis
            x = ((x / cam_range) * self.width // 2 * 0.9) + self.width // 2
            y = ((y / cam_range) * self.height // 2 * 0.9) + self.height // 2
            return (int(x), int(y))
        
        # Draw communication links between agents within r_c range
        r_c = self.world.r_c if hasattr(self.world, 'r_c') else 0.3
        for i, agent_i in enumerate(self.world.agents):
            for j, agent_j in enumerate(self.world.agents):
                if i >= j:  # Only draw each pair once
                    continue
                
                # Calculate distance
                dist = np.linalg.norm(agent_i.state.p_pos - agent_j.state.p_pos)
                
                if dist <= r_c:
                    # Draw dotted line between agents
                    pos_i = world_to_screen(agent_i.state.p_pos)
                    pos_j = world_to_screen(agent_j.state.p_pos)
                    
                    # Draw dotted line by drawing short segments
                    num_segments = 15
                    for seg in range(0, num_segments, 2):  # Skip every other segment for dotted effect
                        t_start = seg / num_segments
                        t_end = (seg + 1) / num_segments
                        
                        x_start = int(pos_i[0] + t_start * (pos_j[0] - pos_i[0]))
                        y_start = int(pos_i[1] + t_start * (pos_j[1] - pos_i[1]))
                        x_end = int(pos_i[0] + t_end * (pos_j[0] - pos_i[0]))
                        y_end = int(pos_i[1] + t_end * (pos_j[1] - pos_i[1]))
                        
                        pygame.draw.line(self.screen, (0, 150, 255), (x_start, y_start), (x_end, y_end), 2)
        
        # Draw agent heading directions as arrows
        for agent in self.world.agents:
            if agent.state.heading is not None:
                pos = world_to_screen(agent.state.p_pos)
                # Arrow length proportional to screen size
                arrow_length = 30
                # Calculate arrow endpoint
                end_x = pos[0] + arrow_length * np.cos(agent.state.heading)
                end_y = pos[1] - arrow_length * np.sin(agent.state.heading)  # negative because y is flipped
                
                # Draw main arrow line
                pygame.draw.line(self.screen, (255, 255, 0), pos, (int(end_x), int(end_y)), 3)
                
                # Draw arrowhead
                arrow_angle = 0.5  # angle of arrowhead in radians
                arrow_head_length = 10
                # Left side of arrowhead
                left_x = end_x - arrow_head_length * np.cos(agent.state.heading - arrow_angle)
                left_y = end_y + arrow_head_length * np.sin(agent.state.heading - arrow_angle)
                pygame.draw.line(self.screen, (255, 255, 0), (int(end_x), int(end_y)), (int(left_x), int(left_y)), 3)
                # Right side of arrowhead
                right_x = end_x - arrow_head_length * np.cos(agent.state.heading + arrow_angle)
                right_y = end_y + arrow_head_length * np.sin(agent.state.heading + arrow_angle)
                pygame.draw.line(self.screen, (255, 255, 0), (int(end_x), int(end_y)), (int(right_x), int(right_y)), 3)
        
        # Draw GPS denied zone as a semi-transparent red rectangle
        if hasattr(self.world, 'gpsd_zone') and len(self.world.gpsd_zone) >= 4:
            # Transform GPS zone corners to screen coordinates
            screen_points = []
            for corner in self.world.gpsd_zone:
                screen_points.append(world_to_screen(corner))
            
            # Draw semi-transparent red rectangle
            surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.polygon(surface, (255, 0, 0, 50), screen_points)  # Red with alpha
            pygame.draw.polygon(surface, (255, 0, 0, 255), screen_points, 3)  # Red border
            self.screen.blit(surface, (0, 0))
            
            # Add label
            self.game_font.render_to(
                self.screen, 
                (self.width * 0.4, self.height * 0.05), 
                "GPS Denied Zone", 
                (255, 0, 0)
            )


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def __init__(self, r_c=0.3, cov_c=0.5):
        self.r_c = r_c        # coverage distance threshold
        self.cov_c = cov_c    # coverage covariance trace threshold
        self.covered = []     # tracks which POIs have been covered

    def make_world(self, N_a=5, cell_width=0.25, speed=1.0, p_noise=0.1, r_c=0.3, r_cov=0.01):
        world = World()
        # No communication action channel (agents are silent);
        # inter-agent info is computed directly in the observation function.
        world.dim_c = 0
        world.dim_p = 2
        world.collaborative = True

        # EKF range measurement parameters
        world.r_c = r_c       # communication radius
        world.r_cov = r_cov   # range measurement noise variance

        # GPS denied zone: center 10x10 square in 20x20 grid
        # Normalized coords: [-0.5, 0.5] within the [-1, 1] world
        world.gpsd_zone = [
            np.array([-0.5, -0.5]),
            np.array([0.5, -0.5]),
            np.array([0.5, 0.5]),
            np.array([-0.5, 0.5]),
        ]
        world.cell_width = cell_width

        # --- agents ---
        world.agents = [Agent() for _ in range(N_a)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = False   # no inter-agent collisions
            agent.silent = True     # no communication action space
            agent.size = 0.04
            agent.p_noise = p_noise # EKF process noise (used by core.update_agent_covariance)
            agent.speed = speed     # constant forward speed

        # --- points of interest (landmarks) ---
        # Discretize the GPS denied zone into a grid with the given cell_width
        # and place a POI at the center of each cell.
        zone_min, zone_max = -0.5, 0.5
        zone_size = zone_max - zone_min
        n_cells = int(np.round(zone_size / cell_width))
        # Recompute actual cell width to fit evenly
        actual_cw = zone_size / n_cells
        # Cell centers: offset by half a cell from zone_min
        centers_1d = zone_min + actual_cw * (np.arange(n_cells) + 0.5)
        # 2D grid of cell centers
        cx, cy = np.meshgrid(centers_1d, centers_1d)
        poi_positions = np.stack([cx.ravel(), cy.ravel()], axis=-1)
        N_r = len(poi_positions)

        world.landmarks = [Landmark() for _ in range(N_r)]
        for i, lm in enumerate(world.landmarks):
            lm.name = f"poi_{i}"
            lm.collide = False
            lm.movable = False
            lm.size = 0.02
            lm.poi_center = poi_positions[i]  # store fixed grid center

        return world

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset_world(self, world, np_random):
        self.covered = [False] * len(world.landmarks)

        # Agent appearance
        for agent in world.agents:
            agent.color = np.array([0.35, 0.35, 0.85])  # blue

        # POI appearance (green = uncovered)
        for lm in world.landmarks:
            lm.color = np.array([0.25, 0.75, 0.25])

        # Place agents as a swarm outside the GPS denied zone
        # GPSD zone center is at (0, 0)
        gpsd_center = np.array([0.0, 0.0])
        
        # Pick a random swarm center location outside GPS denied zone
        swarm_center = None
        while swarm_center is None:
            pos = np_random.uniform(-1, +1, world.dim_p)
            if not world.is_in_gpsd_zone(pos):
                swarm_center = pos
                break
        
        # Place all agents within 0.3 of the swarm center
        for agent in world.agents:
            # Generate random offset within 0.3 radius
            angle = np_random.uniform(0, 2 * np.pi)
            radius = np_random.uniform(0, 0.3)
            offset = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            
            agent.state.p_pos = swarm_center + offset
            # Belief starts at true position (GPS available outside zone)
            agent.state.p_belief = agent.state.p_pos.copy()
            # Constant speed stored as scalar so the unicycle formula works
            agent.state.p_vel = np.array([agent.speed, agent.speed])
            
            # All agents point directly towards the center (0, 0)
            direction_to_center = np.arctan2(gpsd_center[1] - agent.state.p_pos[1], 
                                            gpsd_center[0] - agent.state.p_pos[0])
            agent.state.heading = direction_to_center
            
            agent.state.p_covariance = np.zeros((world.dim_p, world.dim_p))

        # Place POIs at the center of each grid cell (fixed positions)
        for lm in world.landmarks:
            lm.state.p_pos = lm.poi_center.copy()
            lm.state.p_vel = np.zeros(world.dim_p)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_cov_trace(self, agent, world):
        """Return the trace of an agent's position covariance matrix."""
        p_cov = agent.state.p_covariance
        if isinstance(p_cov, np.ndarray) and p_cov.shape == (world.dim_p, world.dim_p):
            return np.trace(p_cov)
        return 0.0

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def reward(self, agent, world):
        """Local per-agent reward: penalise long paths & boundary violations."""
        rew = 0.0

        # Path penalty (each timestep costs a small amount)
        rew -= 0.1

        
        # Add reward proportional to number of agents in communication range
        cov_weight = 0.0
        for other in world.agents:
            if other is agent:
                continue
            dist = np.sqrt(np.sum(np.square(agent.state.p_pos - other.state.p_pos)))
            if dist <= self.r_c and self._get_cov_trace(other, world) < self.cov_c:
                rew+=1.0/(1.0 + self._get_cov_trace(other, world))


        # Penalty for being in GPS denied zone with high covariance
        if world.is_in_gpsd_zone(agent.state.p_pos):
            cov_trace = self._get_cov_trace(agent, world)
            if cov_trace > self.cov_c:
                rew -= float(np.exp(cov_trace - self.cov_c))
        else:
            gpsd_center = np.zeros(world.dim_p)
            dist_to_center = np.linalg.norm(agent.state.p_pos - gpsd_center)
            rew -= 1.0 * np.exp(dist_to_center- 0.5)
            # Penalize moving away from GPS denied zone center
            


        # Boundary penalty – discourage leaving the 20x20 grid
        #for p in range(world.dim_p):
        #    x = abs(agent.state.p_pos[p])
        #    if x > 0.9:


        return rew

    def global_reward(self, world):
        """Global cooperative reward: coverage progress for the whole team."""
        rew = 0.0

        for i, lm in enumerate(world.landmarks):
            if self.covered[i]:
                continue  # already covered

            # Find the closest agent and its covariance
            best_dist = float("inf")
            best_cov_trace = float("inf")
            for agent in world.agents:
                dist = np.sqrt(np.sum(np.square(agent.state.p_pos - lm.state.p_pos)))
                cov_trace = self._get_cov_trace(agent, world)
                if cov_trace < best_cov_trace and dist < self.r_c:
                    best_dist = dist
                    best_cov_trace = cov_trace

            # Check coverage condition
            if best_dist < self.r_c and best_cov_trace < self.cov_c:
                # POI covered! Reward inversely proportional to covariance
                self.covered[i] = True
                lm.color = np.array([0.75, 0.25, 0.25])  # red = covered
                rew += 10.0 / (1.0 + best_cov_trace)
            elif best_dist < self.r_c and best_cov_trace >= self.cov_c:
                # Agent is close to POI but covariance is too high to cover it
                # Penalty inversely proportional to remaining covariance budget
                rew -= float(np.exp(best_cov_trace - self.cov_c))

        # Big bonus reward if all POIs are covered
        if all(self.covered):
            rew += 50.0

        return rew

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------
    def benchmark_data(self, agent, world):
        covered_count = sum(self.covered)
        min_dists = 0
        for i, lm in enumerate(world.landmarks):
            if not self.covered[i]:
                dists = [
                    np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                    for a in world.agents
                ]
                min_dists += min(dists)
        cov_trace = self._get_cov_trace(agent, world)
        return (covered_count, min_dists, cov_trace)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def observation(self, agent, world):
        """Observation vector per agent.

        Layout:
            self_heading        (1)
            self_belief_pos     (2)   -- agent's believed position (not true pos)
            self_cov_trace      (1)
            poi_rel_positions   (N_r * 2)  -- relative to belief
            other_agent_rel_pos ((N_a-1) * 2) -- relative to belief
            other_agent_comm    ((N_a-1) * 2)   [range, cov_trace]
        """
        # Own state (use belief, not true position)
        heading = np.array([agent.state.heading if agent.state.heading is not None else 0.0])
        belief_pos = agent.state.p_belief if agent.state.p_belief is not None else agent.state.p_pos
        cov_trace = np.array([self._get_cov_trace(agent, world)])

        # Relative positions of all points of interest (relative to belief)
        poi_rel_pos = []
        for lm in world.landmarks:
            poi_rel_pos.append(lm.state.p_pos - belief_pos)

        # Other agents: relative position (belief-to-belief) + communication (range, cov_trace)
        other_pos = []
        other_comm = []
        for other in world.agents:
            if other is agent:
                continue
            other_belief = other.state.p_belief if other.state.p_belief is not None else other.state.p_pos
            rel = other_belief - belief_pos
            other_pos.append(rel)
            # Noisy range measurement (only if within communication radius)
            true_dist = np.sqrt(np.sum(np.square(agent.state.p_pos - other.state.p_pos)))
            if true_dist <= world.r_c:
                range_meas = true_dist + np.random.randn() * np.sqrt(world.r_cov)
            else:
                range_meas = -1.0  # Signal: out of range
            other_cov = self._get_cov_trace(other, world)
            other_comm.append(np.array([range_meas, other_cov]))

        return np.concatenate(
            [heading] + [belief_pos] + [cov_trace]
            + poi_rel_pos + other_pos + other_comm
        )
