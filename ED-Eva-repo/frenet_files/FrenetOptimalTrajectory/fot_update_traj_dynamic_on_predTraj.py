import fot_wrapper
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import argparse
from pathlib import Path
import matplotlib.animation as animation
import imageio
from matplotlib import cm

# Define the ego vehicle class
class EgoAgent:
    def __init__(self, init_pos=[0, 0], init_vel=[0, 0], init_ps=0):
        self.position = np.array(init_pos)
        self.velocity = np.array(init_vel)
        self.ps = init_ps
        self.history = []

    def update(self, position, velocity, ps):
        self.position = position
        self.velocity = velocity
        self.ps = ps
        self.history.append(position.copy())
        if len(self.history) > 10:
            self.history.pop(0)

    def draw(self, ax):
        width, length = 2.0, 4.0
        cmap = cm.get_cmap("Reds")
        n = len(self.history) if self.history else 1
        for i, pos in enumerate(self.history):
            color = cmap((i + 1) / n)
            edge_color = tuple(max(0, c - 0.3) for c in color[:3])
            rect = patch.Rectangle((pos[0] - length / 2, pos[1] - width / 2),
                                   length, width,
                                   linewidth=1.5,
                                   edgecolor=edge_color,
                                   facecolor=color)
            ax.add_patch(rect)

# Define moving agent class
class MovingAgent:
    def __init__(self, x_min, y_min, x_max, y_max, vx=1.0, vy=0.5):
        self.width = x_max - x_min
        self.height = y_max - y_min
        self.center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
        self.velocity = np.array([vx, vy])
        self.history = [self.center.copy()]  # initial position history

    def update(self):
        self.center += self.velocity
        self.history.append(self.center.copy())
        if len(self.history) > 20:
            self.history.pop(0)

    def predict_trajectory(self, future_steps=20):
        traj = [(self.center[0] + self.velocity[0] * t,
                 self.center[1] + self.velocity[1] * t) for t in range(future_steps)]
        return traj

    def draw(self, ax):
        # Draw the trail
        cmap = cm.get_cmap("Purples")
        n = len(self.history)
        for i, pos in enumerate(self.history):
            color = cmap((i + 1) / n)
            edge_color = tuple(max(0, c - 0.3) for c in color[:3])
            rect = patch.Rectangle(
                (pos[0] - self.width / 2, pos[1] - self.height / 2),
                self.width, self.height,
                linewidth=1,
                edgecolor=edge_color,
                facecolor=color,
                alpha=0.8
            )
            ax.add_patch(rect)

        # Full predicted trajectory (purple)
        full_traj = self.predict_trajectory()
        traj_x, traj_y = zip(*full_traj)
        ax.plot(traj_x, traj_y, ':', color='purple')

        # Overlay the first 10 steps in red
        used_traj = full_traj[:10]
        used_x, used_y = zip(*used_traj)
        ax.plot(used_x, used_y, ':', color='red', linewidth=2)

# Run fot planner
def fot(show_animation=False,
        show_info=False,
        num_threads=0,
        save_frame=False):
    # Initial conditions
    conds = {
        's0': 0,
        'target_speed': 20,
        'wp': [[0, 0], [50, 0], [150, 0]],
        'obs': [
            [35, -1, 39, 1],
            [48, -2, 52, 2],
                # [98, -4, 102, 2], # no2
                [98, 6, 102, 10],
                [128, 2, 132, 6]],
        'moving_agents': [
            [48, -1, 52, 1, 0.5, 0],
            [98, -1, 102, 1, -0.3, 0.2]
        ],
        'pos': [0, 0],
        'vel': [0, 0],
    }

    initial_conditions = {
        'ps': conds['s0'],
        'target_speed': conds['target_speed'],
        'pos': np.array(conds['pos']),
        'vel': np.array(conds['vel']),
        'wp': np.array(conds['wp']),
        'obs': np.array(conds['obs'])
    }

    hyperparameters = {
        "max_speed": 25.0,
        "max_accel": 15.0,
        "max_curvature": 15.0,
        "max_road_width_l": 5.0,
        "max_road_width_r": 5.0,
        "d_road_w": 0.5,
        "dt": 0.2,
        "maxt": 5.0,
        "mint": 2.0,
        "d_t_s": 0.5,
        "n_s_sample": 2.0,
        "obstacle_clearance": 0.1,
        "kd": 1.0,
        "kv": 0.1,
        "ka": 0.1,
        "kj": 0.1,
        "kt": 0.1,
        "ko": 0.1,
        "klat": 1.0,
        "klon": 1.0,
        "num_threads": num_threads,
    }

    wx = initial_conditions['wp'][:, 0]
    wy = initial_conditions['wp'][:, 1]
    static_obs = np.array(conds['obs'])
    # Create moving agents (visualization only; updated by velocity)
    # moving_agents_list = [MovingAgent(*agent) for agent in conds['moving_agents']]
    common_velocity = [0.4, 0.0]
    moving_agents_list = [
        MovingAgent(48, -1, 52, 1, *common_velocity),
        MovingAgent(98, -2, 102, -4, *common_velocity)
    ]
    # Create ego vehicle instance
    agent = EgoAgent()

    sim_loop = 200
    area = 40
    total_time = 0
    time_list = []
    frames = []  # collect frames for GIF
    fig, ax = plt.subplots() if show_animation else (None, None)

    for i in range(sim_loop):
        print("Iteration: {}".format(i))

        # 1. Update moving agents
        for m_agent in moving_agents_list:
            m_agent.update()

        # 2. Convert moving agents to obstacle boxes: [x_min, y_min, x_max, y_max]
        dynamic_obs = []
        for m_agent in moving_agents_list:
            # Use the predicted trajectory to place a future obstacle box.
            pred_traj = m_agent.predict_trajectory(future_steps=20)
            if len(pred_traj) >= 10:
                x, y = pred_traj[9]  # 10th step (index 9)
                w, h = m_agent.width, m_agent.height
                dynamic_obs.append([x - w/2, y - h/2, x + w/2, y + h/2])


        # 3. Merge static and dynamic obstacles
        all_obs = np.vstack([static_obs, np.array(dynamic_obs)])

        # 4. Update initial_conditions with current obstacles
        initial_conditions['obs'] = all_obs

        start_time = time.time()
        result = fot_wrapper.run_fot(initial_conditions, hyperparameters)
        elapsed_time = time.time() - start_time
        print("Time taken: {}".format(elapsed_time))
        total_time += elapsed_time
        time_list.append(elapsed_time)

        # Validate return format (expects at least 13 elements)
        if not result or len(result) < 13:
            print("Result format error, skipping this iteration.")
            continue

        result_x, result_y, speeds, ix, iy, iyaw, d, s, speeds_x, speeds_y, misc, costs, success = result

        if not success:
            print("Planning failed unexpectedly, skipping this iteration.")
            continue

        # Validate trajectory length
        if len(result_x) < 2 or len(result_y) < 2:
            print("Insufficient trajectory points, skipping this iteration.")
            continue

        new_pos = np.array([result_x[1], result_y[1]])
        new_vel = np.array([speeds_x[1], speeds_y[1]])
        agent.update(new_pos, new_vel, misc['s'])
        initial_conditions['pos'] = new_pos
        initial_conditions['vel'] = new_vel
        initial_conditions['ps'] = misc['s']

        # Stop if the goal is reached (distance to last waypoint <= 3.0)
        if np.hypot(result_x[1] - wx[-1], result_y[1] - wy[-1]) <= 3.0:
            print("Goal reached!")
            break

        if show_animation:
            ax.clear()
            ax.plot(wx, wy, 'k--', label="Waypoints")

            # Draw static obstacles
            for idx, o in enumerate(static_obs):
                rect = patch.Rectangle((o[0], o[1]), o[2] - o[0], o[3] - o[1],
                                    linewidth=0.5, edgecolor='gray', facecolor='gray', alpha=0.7)
                ax.add_patch(rect)

                # Compute obstacle center for labeling
                center_x = (o[0] + o[2]) / 2
                center_y = (o[1] + o[3]) / 2

                # Draw obstacle index above the box
                ax.text(center_x, center_y + 1.5, f'no{idx + 1}', color='black',
                        fontsize=8, ha='center', va='bottom', weight='bold')

            # Draw moving agents (state and predicted trajectories)
            for m_agent in moving_agents_list:
                m_agent.draw(ax)

            ax.plot(result_x[1:], result_y[1:], "-or", label="Trajectory")
            agent.draw(ax)

            ax.set_xlim(result_x[1]-area, result_x[1]+area)
            ax.set_ylim(result_y[1]-area, result_y[1]+area)
            ax.set_xlabel("X axis")
            ax.set_ylabel("Y axis")
            ax.set_title("v[m/s]: " + str(np.linalg.norm(agent.velocity))[:4])
            ax.grid(True)

            if save_frame:
                fig.canvas.draw()
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frames.append(frame)

    if show_animation and save_frame and frames:
        save_path = Path(__file__).resolve().parent.parent / "saves" / "fot_simulation_traj4_check.gif"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(save_path, frames, fps=5)
        print("Saved GIF at: {}".format(save_path))

    print("Finish")
    print("======================= SUMMARY ========================")
    print("Total time for {} iterations taken: {}".format(i+1, total_time))
    print("Average time per iteration: {}".format(total_time/(i+1)))
    print("Max time per iteration: {}".format(max(time_list) if time_list else "N/A"))

    return time_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--display", action="store_true",
                        help="show animation, ensure you have X11 forwarding server open")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="verbose mode, show all state info")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save animation as gif")
    parser.add_argument("-t", "--thread", type=int, default=0,
                        help="set number of threads to run with")
    args = parser.parse_args()

    fot(args.display, args.verbose, args.thread, args.save)
