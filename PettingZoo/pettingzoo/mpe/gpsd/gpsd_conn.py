# noqa: D212, D415
"""
# GPSD-Conn — GPS Denied Coverage with Connectivity Awareness

Extended version of gpsd.py that adds:
  1. Connectivity reward shaping   — buffer-zone penalty  R_spatial
  2. Global connectivity reward    — penalty when Fiedler value drops to 0
  3. Adjacency + Laplacian info    — exposed in `info` dict for the critic
  4. Curriculum support            — adjustable connectivity penalty weight

The observation space is *identical* to the base gpsd.py so that the same
vectorised wrappers work.  The additional graph-topology data is passed
through the `info` dict (one per agent).

Usage:
    from pettingzoo.mpe.gpsd.gpsd_conn import parallel_env
"""

import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn


# ======================================================================
# Topology helpers (pure numpy – no external deps)
# ======================================================================

def build_adjacency(positions, r_c):
    """Build binary adjacency matrix from agent positions.

    Args:
        positions: (N, 2) array of agent positions.
        r_c: communication radius.

    Returns:
        adj: (N, N) binary adjacency matrix (0 on diagonal).
    """
    N = len(positions)
    adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(positions[i] - positions[j])
            if d <= r_c:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
    return adj


def graph_laplacian(adj):
    """Compute the combinatorial graph Laplacian L = D - A."""
    D = np.diag(adj.sum(axis=1))
    return D - adj


def fiedler_value(adj):
    """Algebraic connectivity = second-smallest eigenvalue of the Laplacian.

    Returns 0.0 when the graph is disconnected; >0 when connected.
    """
    L = graph_laplacian(adj)
    eigvals = np.linalg.eigvalsh(L)  # sorted ascending
    if len(eigvals) < 2:
        return 0.0
    return float(max(eigvals[1], 0.0))  # clamp numerics


# ======================================================================
# raw_env  (extends SimpleEnv exactly like gpsd.py)
# ======================================================================

class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        N_a=5,
        cell_width=0.25,
        local_ratio=0.1,
        max_cycles=200,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
        min_w=-1.0,
        max_w=1.0,
        speed=0.1,
        r_c=0.3,
        cov_c=0.5,
        p_noise=0.1,
        r_cov=0.01,
        # --- new knobs ---
        conn_penalty_weight=1.0,   # weight for connectivity reward shaping
        buffer_eps=0.05,           # buffer zone ε for R_spatial
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
            conn_penalty_weight=conn_penalty_weight,
            buffer_eps=buffer_eps,
        )
        assert 0.0 <= local_ratio <= 1.0

        self.omega_values = np.linspace(min_w, max_w, 5)
        self.conn_penalty_weight = conn_penalty_weight
        self.buffer_eps = buffer_eps

        scenario = Scenario(r_c=r_c, cov_c=cov_c)
        world = scenario.make_world(
            N_a, cell_width=cell_width, speed=speed,
            p_noise=p_noise, r_c=r_c, r_cov=r_cov,
        )
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
        self.metadata["name"] = "gpsd_conn_v1"

    # ------------------------------------------------------------------
    def _execute_world_step(self):
        """Override: inject connectivity metrics + reward shaping."""
        super()._execute_world_step()

        # --- Build topology ---
        positions = np.array([a.state.p_pos for a in self.world.agents])
        r_c = self.world.r_c if hasattr(self.world, 'r_c') else 0.3
        adj = build_adjacency(positions, r_c)
        lap = graph_laplacian(adj)
        fv = fiedler_value(adj)

        # Coverage ratio
        if hasattr(self.scenario, 'covered') and self.scenario.covered:
            coverage_ratio = sum(self.scenario.covered) / len(self.scenario.covered)
        else:
            coverage_ratio = 0.0

        # --- Buffer-zone spatial penalty (R_spatial) ---
        # -sum max(0, dist(i,j) - (R - ε))  for all agent pairs
        eps = self.buffer_eps
        N = len(self.world.agents)
        spatial_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(positions[i] - positions[j])
                excess = max(0.0, d - (r_c - eps))
                spatial_penalty += excess
        spatial_penalty *= -self.conn_penalty_weight

        # --- Connectivity reward: 0 if connected, -λ if partitioned ---
        conn_reward = 0.0 if fv > 1e-6 else -self.conn_penalty_weight

        # --- Inject into info dicts ---
        for agent_name in self.agents:
            self.infos[agent_name]["coverage_ratio"] = coverage_ratio
            self.infos[agent_name]["adjacency"] = adj.copy()
            self.infos[agent_name]["laplacian"] = lap.copy()
            self.infos[agent_name]["fiedler_value"] = fv
            self.infos[agent_name]["spatial_penalty"] = spatial_penalty
            self.infos[agent_name]["conn_reward"] = conn_reward

        # --- Add connectivity shaping to rewards ---
        for agent_name in self.agents:
            if agent_name in self.rewards:
                self.rewards[agent_name] += spatial_penalty / N + conn_reward / N

    # ------------------------------------------------------------------
    def _set_action(self, action, agent, action_space, time=None):
        """Map discrete action index → scalar omega (identical to gpsd.py)."""
        agent.action.c = np.zeros(self.world.dim_c)
        if agent.movable:
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

    # ------------------------------------------------------------------
    def draw(self):
        """Override draw to add GPS denied zone visualization and communication links."""
        import pygame

        self.scenario._update_agent_colors(self.world)
        self.screen.fill((255, 255, 255))
        cam_range = 1.1

        def world_to_screen(pos):
            x, y = pos
            y *= -1
            x = ((x / cam_range) * self.width // 2 * 0.9) + self.width // 2
            y = ((y / cam_range) * self.height // 2 * 0.9) + self.height // 2
            return (int(x), int(y))

        for entity in self.world.entities:
            x, y = world_to_screen(entity.state.p_pos)
            radius = int(entity.size * 350)
            pygame.draw.circle(self.screen, entity.color * 200, (x, y), radius)
            pygame.draw.circle(self.screen, (0, 0, 0), (x, y), radius, 1)

        r_c = self.world.r_c if hasattr(self.world, 'r_c') else 0.3
        for i, agent_i in enumerate(self.world.agents):
            for j, agent_j in enumerate(self.world.agents):
                if i >= j:
                    continue
                dist = np.linalg.norm(agent_i.state.p_pos - agent_j.state.p_pos)
                if dist <= r_c:
                    pos_i = world_to_screen(agent_i.state.p_pos)
                    pos_j = world_to_screen(agent_j.state.p_pos)
                    num_segments = 15
                    for seg in range(0, num_segments, 2):
                        t_start = seg / num_segments
                        t_end = (seg + 1) / num_segments
                        x_start = int(pos_i[0] + t_start * (pos_j[0] - pos_i[0]))
                        y_start = int(pos_i[1] + t_start * (pos_j[1] - pos_i[1]))
                        x_end = int(pos_i[0] + t_end * (pos_j[0] - pos_i[0]))
                        y_end = int(pos_i[1] + t_end * (pos_j[1] - pos_i[1]))
                        pygame.draw.line(self.screen, (0, 150, 255),
                                         (x_start, y_start), (x_end, y_end), 2)

        for agent in self.world.agents:
            if agent.state.heading is not None:
                pos = world_to_screen(agent.state.p_pos)
                arrow_length = 30
                end_x = pos[0] + arrow_length * np.cos(agent.state.heading)
                end_y = pos[1] - arrow_length * np.sin(agent.state.heading)
                pygame.draw.line(self.screen, (255, 255, 0), pos,
                                 (int(end_x), int(end_y)), 3)
                arrow_angle = 0.5
                arrow_head_length = 10
                left_x = end_x - arrow_head_length * np.cos(agent.state.heading - arrow_angle)
                left_y = end_y + arrow_head_length * np.sin(agent.state.heading - arrow_angle)
                pygame.draw.line(self.screen, (255, 255, 0),
                                 (int(end_x), int(end_y)), (int(left_x), int(left_y)), 3)
                right_x = end_x - arrow_head_length * np.cos(agent.state.heading + arrow_angle)
                right_y = end_y + arrow_head_length * np.sin(agent.state.heading + arrow_angle)
                pygame.draw.line(self.screen, (255, 255, 0),
                                 (int(end_x), int(end_y)), (int(right_x), int(right_y)), 3)

        if hasattr(self.world, 'gpsd_zone') and len(self.world.gpsd_zone) >= 4:
            screen_points = [world_to_screen(c) for c in self.world.gpsd_zone]
            surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.polygon(surface, (255, 0, 0, 50), screen_points)
            pygame.draw.polygon(surface, (255, 0, 0, 255), screen_points, 3)
            self.screen.blit(surface, (0, 0))
            self.game_font.render_to(
                self.screen, (self.width * 0.4, self.height * 0.05),
                "GPS Denied Zone", (255, 0, 0),
            )


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


# ======================================================================
# Scenario  (identical to gpsd.py – copied to keep the file self-contained)
# ======================================================================

class Scenario(BaseScenario):

    def __init__(self, r_c=0.3, cov_c=0.5):
        self.r_c = r_c
        self.cov_c = cov_c
        self.covered = []

    def make_world(self, N_a=5, cell_width=0.25, speed=1.0,
                   p_noise=0.1, r_c=0.3, r_cov=0.01):
        world = World()
        world.dim_c = 0
        world.dim_p = 2
        world.collaborative = True
        world.r_c = r_c
        world.r_cov = r_cov

        world.gpsd_zone = [
            np.array([-0.5, -0.5]),
            np.array([0.5, -0.5]),
            np.array([0.5, 0.5]),
            np.array([-0.5, 0.5]),
        ]
        world.cell_width = cell_width

        world.agents = [Agent() for _ in range(N_a)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = False
            agent.silent = True
            agent.size = 0.04
            agent.p_noise = p_noise
            agent.speed = speed

        zone_min, zone_max = -0.5, 0.5
        zone_size = zone_max - zone_min
        n_cells = int(np.round(zone_size / cell_width))
        world.n_cells = n_cells
        actual_cw = zone_size / n_cells
        centers_1d = zone_min + actual_cw * (np.arange(n_cells) + 0.5)
        cx, cy = np.meshgrid(centers_1d, centers_1d)
        poi_positions = np.stack([cx.ravel(), cy.ravel()], axis=-1)
        N_r = len(poi_positions)

        world.landmarks = [Landmark() for _ in range(N_r)]
        for i, lm in enumerate(world.landmarks):
            lm.name = f"poi_{i}"
            lm.collide = False
            lm.movable = False
            lm.size = 0.02
            lm.poi_center = poi_positions[i]

        return world

    # ------------------------------------------------------------------
    USE_LINE_SPAWN = False

    def reset_world(self, world, np_random):
        self.covered = [False] * len(world.landmarks)
        for agent in world.agents:
            agent.color = np.array([0.35, 0.35, 0.85])
        for lm in world.landmarks:
            lm.color = np.array([0.25, 0.75, 0.25])
            lm.covered = False
        if self.USE_LINE_SPAWN:
            self._spawn_line(world, np_random)
        else:
            self._spawn_cluster(world, np_random)
        for lm in world.landmarks:
            lm.state.p_pos = lm.poi_center.copy()
            lm.state.p_vel = np.zeros(world.dim_p)

    # ------------------------------------------------------------------
    def _pick_swarm_center(self, world, np_random, radius=0.75):
        while True:
            angle = np_random.uniform(0, 2 * np.pi)
            pos = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            if not world.is_in_gpsd_zone(pos):
                approach_angle = np.arctan2(-pos[1], -pos[0])
                return pos, approach_angle

    def _init_agent_state(self, agent, pos, heading, world):
        agent.state.p_pos = pos
        agent.state.p_belief = pos.copy()
        agent.state.p_vel = np.array([agent.speed, agent.speed])
        agent.state.heading = heading
        agent.state.p_covariance = np.zeros((world.dim_p, world.dim_p))

    def _spawn_cluster(self, world, np_random):
        swarm_center, approach_angle = self._pick_swarm_center(world, np_random)
        for agent in world.agents:
            r = np_random.uniform(0, self.r_c)
            theta = np_random.uniform(0, 2 * np.pi)
            offset = np.array([r * np.cos(theta), r * np.sin(theta)])
            pos = swarm_center + offset
            noisy_angle = approach_angle + np_random.uniform(-np.pi / 2, np.pi / 2)
            self._init_agent_state(agent, pos, noisy_angle, world)

    def _spawn_line(self, world, np_random):
        swarm_center, approach_angle = self._pick_swarm_center(world, np_random)
        n = len(world.agents)
        spacing = 0.1
        perp_angle = approach_angle + np.pi / 2
        perp_dir = np.array([np.cos(perp_angle), np.sin(perp_angle)])
        offsets = (np.arange(n) - (n - 1) / 2.0) * spacing
        for i, agent in enumerate(world.agents):
            pos = swarm_center + offsets[i] * perp_dir
            self._init_agent_state(agent, pos, approach_angle, world)

    # ------------------------------------------------------------------
    def _get_cov_trace(self, agent, world):
        p_cov = agent.state.p_covariance
        if isinstance(p_cov, np.ndarray) and p_cov.shape == (world.dim_p, world.dim_p):
            return np.trace(p_cov)
        return 0.0

    def _update_agent_colors(self, world):
        cov_threshold = self.cov_c
        for agent in world.agents:
            cov_trace = self._get_cov_trace(agent, world)
            if cov_trace < cov_threshold:
                safety = cov_threshold - cov_trace
                decay_rate = 20.0
                decay_factor = np.exp(-decay_rate * safety)
                agent.color = (np.array([0.25, 0.75, 0.25]) * decay_factor
                               + np.array([0.35, 0.35, 0.85]) * (1 - decay_factor))
            else:
                excess = cov_trace - cov_threshold
                decay_factor = np.exp(-20.0 * excess)
                agent.color = np.array([0.8, 0.4, 0.8]) * decay_factor

    def _world_to_body_frame(self, rel_pos_world, heading):
        dx, dy = rel_pos_world
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        x_body = sin_h * dx - cos_h * dy
        y_body = cos_h * dx + sin_h * dy
        return np.array([x_body, y_body])

    # ------------------------------------------------------------------
    # Reward  (identical to gpsd.py)
    # ------------------------------------------------------------------
    def reward(self, agent, world):
        rew = 0.0
        connections = 0
        for other in world.agents:
            if other is agent:
                continue
            dist = np.sqrt(np.sum(np.square(agent.state.p_pos - other.state.p_pos)))
            if dist <= self.r_c and (world.is_in_gpsd_zone(other.state.p_pos)
                                     or world.is_in_gpsd_zone(agent.state.p_pos)):
                rew += 0.125 / (1.0 + 10 * self._get_cov_trace(other, world))
            if dist <= self.r_c and world.is_in_gpsd_zone(other.state.p_pos):
                connections += 1

        cov_trace = self._get_cov_trace(agent, world)
        if world.is_in_gpsd_zone(agent.state.p_pos):
            rew += 0.1 * connections / (1.0 + cov_trace)
        else:
            gpsd_center = np.zeros(world.dim_p)
            dist_to_center = np.linalg.norm(agent.state.p_pos - gpsd_center)
            if connections == 0:
                rew -= 0.5 * dist_to_center
            else:
                rew += 1.0 * connections / (1.0 + cov_trace)
        return rew

    def global_reward(self, world):
        rew = 0.0
        rew -= 0.01
        for i, lm in enumerate(world.landmarks):
            if self.covered[i]:
                continue
            for agent in world.agents:
                dist = np.linalg.norm(agent.state.p_pos - lm.state.p_pos)
                rew += 0.02 * max(0.0, 1.0 - dist / self.r_c)

        total_pois = len(world.landmarks)
        num_covered = sum(self.covered)

        for i, lm in enumerate(world.landmarks):
            if self.covered[i]:
                continue
            best_dist = float("inf")
            best_cov_trace = float("inf")
            for agent in world.agents:
                dist = np.sqrt(np.sum(np.square(agent.state.p_pos - lm.state.p_pos)))
                cov_trace = self._get_cov_trace(agent, world)
                if cov_trace < best_cov_trace and dist < self.r_c:
                    best_dist = dist
                    best_cov_trace = cov_trace
            if best_dist < self.r_c and best_cov_trace < self.cov_c:
                base_reward = 1.0 / (1.0 + best_cov_trace)
                coverage_ratio = num_covered / total_pois
                scale_factor = 1.0 + 9.0 * coverage_ratio
                self.covered[i] = True
                lm.covered = True
                lm.color = np.array([0.75, 0.25, 0.25])
                rew += base_reward * scale_factor
                num_covered += 1
            elif best_dist < self.r_c and best_cov_trace < float("inf"):
                rew += 0.05 * (1.0 - min(best_cov_trace / self.cov_c, 1.0))

        if all(self.covered):
            rew += 5.0
        return rew

    # ------------------------------------------------------------------
    def benchmark_data(self, agent, world):
        covered_count = sum(self.covered)
        min_dists = 0
        for i, lm in enumerate(world.landmarks):
            if not self.covered[i]:
                dists = [np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                         for a in world.agents]
                min_dists += min(dists)
        cov_trace = self._get_cov_trace(agent, world)
        return (covered_count, min_dists, cov_trace)

    # ------------------------------------------------------------------
    # Observation (identical to gpsd.py)
    # ------------------------------------------------------------------
    def observation(self, agent, world):
        heading_val = agent.state.heading if agent.state.heading is not None else 0.0
        heading = np.array([heading_val])
        belief_pos = (agent.state.p_belief
                      if agent.state.p_belief is not None else agent.state.p_pos)
        cov_trace = np.array([self._get_cov_trace(agent, world)])
        in_gpsd_zone = np.array([float(world.is_in_gpsd_zone(belief_pos))])

        poi_rel_pos = []
        for lm in world.landmarks:
            rel_world = lm.state.p_pos - belief_pos
            rel_body = self._world_to_body_frame(rel_world, heading_val)
            if np.linalg.norm(rel_body) < world.r_c:
                poi_rel_pos.append(rel_body)
            else:
                poi_rel_pos.append(np.array([-1.0, -1.0]))

        other_pos = []
        other_comm = []
        other_heading = []
        for other in world.agents:
            if other is agent:
                continue
            other_belief = (other.state.p_belief
                            if other.state.p_belief is not None else other.state.p_pos)
            rel_world = other_belief - belief_pos
            rel_body = self._world_to_body_frame(rel_world, heading_val)
            rel_heading = np.abs(
                np.arctan2(rel_body[1], rel_body[0]) - heading
                if rel_body[0] != 0 else 0.0
            )
            true_dist = np.sqrt(np.sum(np.square(
                agent.state.p_pos - other.state.p_pos
            )))
            other_cov = self._get_cov_trace(other, world)
            if true_dist <= 10 * world.r_c:
                range_meas = true_dist + np.random.randn() * np.sqrt(world.r_cov)
                other_pos.append(rel_body)
                other_heading.append(rel_heading)
            else:
                range_meas = -1.0
                other_cov = -1.0
                other_pos.append(np.array([-1.0, -1.0]))
                other_heading.append(np.array([-1.0]))
            other_comm.append(np.array([range_meas, other_cov]))

        return np.concatenate(
            [heading] + [belief_pos] + [cov_trace] + [in_gpsd_zone]
            + poi_rel_pos + other_pos + other_comm + other_heading
        )
