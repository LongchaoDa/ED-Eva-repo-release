import fot_wrapper
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import argparse
from pathlib import Path
import imageio.v2 as imageio


# Run fot planner
def fot(show_animation=False,
        show_info=False,
        num_threads=0,
        save_frame=False):
    conds = {
        's0': 0,
        'target_speed': 20,
        'wp': [[0, 0], [50, 0], [150, 0]],
        'obs': [[48, -2, 52, 2], [98, -4, 102, 2], [98, 6, 102, 10],
                [128, 2, 132, 6]],
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
    obs = np.array(conds['obs'])

    sim_loop = 200
    area = 40
    total_time = 0
    time_list = []

    output_dir = Path("/home/ed-eva/frenet_optimal_trajectory_planner/output")
    frame_dir = output_dir / "frames"
    frames = []

    if save_frame:
        frame_dir.mkdir(parents=True, exist_ok=True)

    for i in range(sim_loop):
        print("Iteration: {}".format(i))
        start_time = time.time()
        result_x, result_y, speeds, ix, iy, iyaw, d, s, speeds_x, \
            speeds_y, misc, costs, success = \
            fot_wrapper.run_fot(initial_conditions, hyperparameters)
        end_time = time.time() - start_time
        print("Time taken: {}".format(end_time))
        total_time += end_time
        time_list.append(end_time)

        if success:
            initial_conditions['pos'] = np.array([result_x[1], result_y[1]])
            initial_conditions['vel'] = np.array([speeds_x[1], speeds_y[1]])
            initial_conditions['ps'] = misc['s']
            if show_info:
                print(costs)
        else:
            print("Failed unexpectedly")
            break

        if np.hypot(result_x[1] - wx[-1], result_y[1] - wy[-1]) <= 3.0:
            print("Goal")
            if show_animation or save_frame:
                plt.cla()
                plt.plot(wx, wy)
                if obs.shape[0] == 0:
                    obs = np.empty((0, 4))
                ax = plt.gca()
                for o in obs:
                    rect = patch.Rectangle((o[0], o[1]), o[2] - o[0], o[3] - o[1])
                    ax.add_patch(rect)
                plt.plot(result_x[1:], result_y[1:], "-or")
                plt.plot(result_x[1], result_y[1], "vc")
                plt.xlim(result_x[1] - area, result_x[1] + area)
                plt.ylim(result_y[1] - area, result_y[1] + area)
                plt.xlabel("X axis")
                plt.ylabel("Y axis")
                plt.title("v[m/s]:" +
                          str(np.linalg.norm(initial_conditions['vel']))[0:4])
                plt.grid(True)
                if save_frame:
                    frame_path = frame_dir / f"{i:04d}.png"
                    plt.savefig(frame_path, dpi=120)
                    frames.append(frame_path)
            break

        if show_animation or save_frame:
            plt.cla()
            if show_animation:  # pragma: no cover
                plt.gcf().canvas.mpl_connect(
                    "key_release_event",
                    lambda event: [exit(0) if event.key == "escape" else None]
                )
            plt.plot(wx, wy)
            if obs.shape[0] == 0:
                obs = np.empty((0, 4))
            ax = plt.gca()
            for o in obs:
                rect = patch.Rectangle((o[0], o[1]), o[2] - o[0], o[3] - o[1])
                ax.add_patch(rect)
            plt.plot(result_x[1:], result_y[1:], "-or")
            plt.plot(result_x[1], result_y[1], "vc")
            plt.xlim(result_x[1] - area, result_x[1] + area)
            plt.ylim(result_y[1] - area, result_y[1] + area)
            plt.xlabel("X axis")
            plt.ylabel("Y axis")
            plt.title("v[m/s]:" +
                      str(np.linalg.norm(initial_conditions['vel']))[0:4])
            plt.grid(True)

            if save_frame:
                frame_path = frame_dir / f"{i:04d}.png"
                plt.savefig(frame_path, dpi=120)
                frames.append(frame_path)

            if show_animation:  # pragma: no cover
                plt.pause(0.1)

    print("Finish")
    print("======================= SUMMARY ========================")
    print("Total time for {} iterations taken: {}".format(i, total_time))
    if i > 0:
        print("Average time per iteration: {}".format(total_time / i))
    else:
        print("Average time per iteration: N/A")
    if len(time_list) > 0:
        print("Max time per iteration: {}".format(max(time_list)))
    else:
        print("Max time per iteration: N/A")

    if save_frame and len(frames) > 0:
        gif_path = output_dir / "frenet_simulation.gif"
        images = [imageio.imread(frame) for frame in frames]
        imageio.mimsave(gif_path, images, duration=0.1, loop=0)
        print(f"GIF saved to: {gif_path}")

    return time_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--display",
        action="store_true",
        help="show animation, ensure you have X11 forwarding server open")
    parser.add_argument("-v",
                        "--verbose",
                        action="store_true",
                        help="verbose mode, show all state info")
    parser.add_argument("-s",
                        "--save",
                        action="store_true",
                        help="save each frame and export GIF")
    parser.add_argument("-t",
                        "--thread",
                        type=int,
                        default=0,
                        help="set number of threads to run with")
    args = parser.parse_args()

    fot(args.display, args.verbose, args.thread, args.save)