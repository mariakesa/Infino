import numpy as np

class BouncingBallWorld:
    def __init__(self, 
                 gravity=-9.8, 
                 dt=0.05, 
                 initial_position=1.0, 
                 initial_velocity=0.0, 
                 floor=0.0, 
                 ceiling=2.0,
                 restitution=0.9):
        """
        Simulates 1D vertical motion of a bouncing ball under gravity.
        """
        self.g = gravity
        self.dt = dt
        self.floor = floor
        self.ceiling = ceiling
        self.restitution = restitution

        self.reset(initial_position, initial_velocity)

    def reset(self, x=None, v=None):
        self.x = x if x is not None else np.random.uniform(self.floor, self.ceiling)
        self.v = v if v is not None else 0.0
        self.t = 0.0
        self.history = {"t": [], "x": [], "v": []}

    def step(self):
        # Update velocity and position
        self.v += self.g * self.dt
        self.x += self.v * self.dt
        self.t += self.dt

        # Check for collision with floor
        if self.x <= self.floor:
            self.x = self.floor
            self.v = -self.v * self.restitution

        # Check for collision with ceiling
        if self.x >= self.ceiling:
            self.x = self.ceiling
            self.v = -self.v * self.restitution

        # Save state
        self.history["t"].append(self.t)
        self.history["x"].append(self.x)
        self.history["v"].append(self.v)

    def simulate(self, n_steps=100):
        self.reset(self.x, self.v)
        for _ in range(n_steps):
            self.step()
        return self.history
