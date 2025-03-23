import matplotlib.pyplot as plt
from infino.world.bouncing_ball import BouncingBallWorld

world = BouncingBallWorld()
trajectory = world.simulate(n_steps=200)

plt.plot(trajectory["t"], trajectory["x"])
plt.title("Bouncing Ball Trajectory")
plt.xlabel("Time")
plt.ylabel("Height")
plt.grid()
plt.show()