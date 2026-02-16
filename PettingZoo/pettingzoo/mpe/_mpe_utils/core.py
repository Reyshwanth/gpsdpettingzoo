import numpy as np


class EntityState:  # physical/external base state of all entities
    def __init__(self):
        # physical position
        self.p_pos = None
        self.heading = None
        # physical velocity
        self.p_vel = None


class AgentState(
    EntityState
):  # state of agents (including communication and internal/mental state)
    def __init__(self):
        super().__init__()
        # position belief (estimated position, may differ from true p_pos)
        self.p_belief = None
        # position error covariance matrix
        self.p_covariance = None


class Action:  # action of the agent
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None


class Entity:  # properties and state of physical world entity
    def __init__(self):
        # name
        self.name = ""
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = False
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass


class Landmark(Entity):  # properties of landmark entities
    def __init__(self):
        super().__init__()


class Agent(Entity):  # properties of agent entities
    def __init__(self):
        super().__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None


class World:  # multi-agent world
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.gpsd_zone = []  # 4 points defining the GPS denied Zone
        # communication channel dimensionality
        self.dim_c = 2  # range measurement and error covariance
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e2
        self.contact_margin = 1e-3
        # EKF range measurement parameters
        self.r_c = 0.3       # communication radius for range measurements
        self.r_cov = 0.01    # range measurement noise variance (sigma^2)

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_omega = [None] * len(self.entities)
        # apply agent physical controls
        p_omega = self.apply_action_omega(p_omega)
        # apply environment forces
        # integrate physical state
        self.integrate_state(p_omega)
        # update EKF prediction (covariance) for each agent
        for agent in self.agents:
            self.update_agent_covariance(agent)
        # update EKF with range measurements from neighbours
        self.update_ekf_range_measurements()
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_omega(self, p_omega):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = (
                    np.random.randn(*agent.action.u.shape) * agent.u_noise
                    if agent.u_noise
                    else 0.0
                )
                p_omega[i] = agent.action.u + noise
        return p_omega

    # gather physical forces acting on entities


    # integrate physical state
    def integrate_state(self, p_omega):
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            # Compute position delta from unicycle kinematics
            if p_omega[i] is not None and abs(p_omega[i]) > 1e-8:
                dp = (
                    (entity.state.p_vel / p_omega[i])
                    * np.array(
                        [
                            np.sin(entity.state.heading + p_omega[i] * self.dt)
                            - np.sin(entity.state.heading),
                            -np.cos(entity.state.heading + p_omega[i] * self.dt)
                            + np.cos(entity.state.heading),
                        ]
                    )
                )
            else:
                dp = (
                    entity.state.p_vel
                    * self.dt
                    * np.array([np.cos(entity.state.heading), np.sin(entity.state.heading)])
                )
            
            # Update true position
            entity.state.p_pos += dp
            
            # Propagate belief with the same kinematics (dead reckoning)
            if hasattr(entity.state, 'p_belief') and entity.state.p_belief is not None:
                entity.state.p_belief += dp
            
            entity.state.heading += p_omega[i] * self.dt

    def is_in_gpsd_zone(self, position):
        """Check if a position is inside the GPS denied zone using ray casting algorithm."""
        if not self.gpsd_zone or len(self.gpsd_zone) < 3:
            return False  # No valid GPS denied zone defined
        
        x, y = position
        n = len(self.gpsd_zone)
        inside = False
        
        p1x, p1y = self.gpsd_zone[0]
        for i in range(1, n + 1):
            p2x, p2y = self.gpsd_zone[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside

    def update_agent_covariance(self, agent):
        """Update agent position covariance using EKF prediction step."""
        # Check if agent is in GPS denied zone
        in_gpsd_zone = self.is_in_gpsd_zone(agent.state.p_pos)
        
        if in_gpsd_zone:
            # Agent is in GPS denied zone - update covariance with EKF prediction
            # Get process noise for this agent
            p_noise = agent.p_noise if agent.p_noise is not None else 0.0
            noise_vec = np.atleast_1d(p_noise).astype(float)
            
            # Build process noise covariance matrix Q
            if noise_vec.size == 1:
                q = (noise_vec[0] ** 2) * np.eye(self.dim_p)
            else:
                q = np.diag(noise_vec[: self.dim_p] ** 2)
            
            # Get current covariance or initialize if needed
            p_cov = agent.state.p_covariance
            if not isinstance(p_cov, np.ndarray) or p_cov.shape != (self.dim_p, self.dim_p):
                p_cov = np.zeros((self.dim_p, self.dim_p))
            
            # EKF prediction step: P_k+1 = F * P_k * F^T + Q
            # F is identity for simple constant velocity model
            f = np.eye(self.dim_p)
            agent.state.p_covariance = f @ p_cov @ f.T + q
        else:
            # Agent has GPS - position is known with certainty
            agent.state.p_covariance = np.zeros((self.dim_p, self.dim_p))
            # Reset belief to true position when GPS is available
            agent.state.p_belief = agent.state.p_pos.copy()

    def update_ekf_range_measurements(self):
        """EKF measurement update using range measurements from agents within r_c.

        For each pair of agents (i, j) within communication radius r_c:
        - A noisy range measurement z = ||p_pos_i - p_pos_j|| + noise is taken
        - Agent i uses the measurement to update its belief and covariance
        - The measurement model: h(x) = ||x_belief_i - x_belief_j||
        - Jacobian H_i = (x_belief_i - x_belief_j)^T / predicted_range
        - Innovation covariance accounts for both agents' uncertainties + range noise
        """
        n = len(self.agents)
        if n < 2:
            return

        # Collect current beliefs and covariances (snapshot before updates)
        beliefs = []
        covs = []
        for agent in self.agents:
            b = agent.state.p_belief
            if b is None or not isinstance(b, np.ndarray):
                b = agent.state.p_pos.copy()
            beliefs.append(b.copy())

            p_cov = agent.state.p_covariance
            if not isinstance(p_cov, np.ndarray) or p_cov.shape != (self.dim_p, self.dim_p):
                p_cov = np.zeros((self.dim_p, self.dim_p))
            covs.append(p_cov.copy())

        # Range measurement noise covariance (scalar → 1x1 matrix)
        R = np.array([[self.r_cov]])

        # Process each pair
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                # True range between agents (used to generate noisy measurement)
                true_range = np.sqrt(np.sum(np.square(
                    self.agents[i].state.p_pos - self.agents[j].state.p_pos
                )))

                # Only communicate if within range
                if true_range > self.r_c:
                    continue

                # --- Generate noisy range measurement ---
                z = true_range + np.random.randn() * np.sqrt(self.r_cov)
                z = max(z, 1e-8)  # Ensure positive range

                # --- Predicted range from beliefs ---
                delta_belief = beliefs[i] - beliefs[j]
                predicted_range = np.sqrt(np.sum(np.square(delta_belief)))

                if predicted_range < 1e-8:
                    # Agents' beliefs are co-located; skip to avoid division by zero
                    continue

                # --- Jacobian of h(x) = ||x_i - x_j|| w.r.t. x_i ---
                # H_i is (1 x dim_p):  (x_i - x_j)^T / ||x_i - x_j||
                H_i = (delta_belief / predicted_range).reshape(1, self.dim_p)

                # Jacobian w.r.t. x_j is -H_i (for innovation covariance)
                H_j = -H_i

                # --- Innovation covariance ---
                # S = H_i @ P_i @ H_i^T + H_j @ P_j @ H_j^T + R
                # This accounts for uncertainty in both agents' positions + measurement noise
                S = H_i @ covs[i] @ H_i.T + H_j @ covs[j] @ H_j.T + R
                S_inv = 1.0 / S[0, 0] if S[0, 0] > 1e-12 else 0.0

                # --- Innovation (measurement residual) ---
                y = z - predicted_range

                # --- Kalman gain ---
                # K_i = P_i @ H_i^T @ S^{-1}   shape: (dim_p, 1)
                K = covs[i] @ H_i.T * S_inv

                # --- Update belief ---
                beliefs[i] = beliefs[i] + (K * y).flatten()

                # --- Update covariance ---
                # P_i = (I - K @ H_i) @ P_i
                I = np.eye(self.dim_p)
                covs[i] = (I - K @ H_i) @ covs[i]

        # Write back updated beliefs and covariances
        for i, agent in enumerate(self.agents):
            agent.state.p_belief = beliefs[i]
            agent.state.p_covariance = covs[i]

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = (
                np.random.randn(*agent.action.c.shape) * agent.c_noise
                if agent.c_noise
                else 0.0
            )
            agent.state.c = agent.action.c + noise

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if entity_a is entity_b:
            return [None, None]  # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]
