import os
from typing import List

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from shapely.geometry import Polygon

from vslam2tag.utils.dataset_utilities import get_marker_dict
from vslam2tag.utils.definitions import get_project_root, LANDMARK_COL

root = get_project_root()

FLOOR = 4
DATA_PATH = "/data/"
EVALUATION_PATH = "/evaluation/evaluation_data/"


class FloorplanPlot():
    def __init__(self, floorplan_dimensions, grid_size, draw_grid=False, floorplan_bg_img="",
                 filename=None, sample_points_file=None, add_points=False, delete_points=False,
                 walls_file=None, add_walls=False, xtick_freq=None, correct_walls=False, title=None,
                 walls=None, axis=None, background_alpha=0.3):

        self.floorplan_dimensions = floorplan_dimensions
        self.grid_size = grid_size
        self.filename = filename
        self.floorplan_bg_image = floorplan_bg_img

        self.axis: Axes = axis
        self.fig: Figure = None
        self.sample_points_file = sample_points_file
        self.delete_points = delete_points
        self.delete_th = 0.5
        self.sample_points = None

        self.start_wall_point = None
        self.new_walls = None
        self.walls_file = walls_file
        self.nav_bar = None
        self.bg_img: AxesImage = None
        self.bg_alpha = background_alpha
        self.draw_background()

        if title is not None:
            self.set_title(title)
        if draw_grid:
            self.draw_grid()
        if xtick_freq is not None:
            self.set_tick_frequency(xtick_freq)

        if sample_points_file is not None:
            if os.path.exists(sample_points_file):
                self.sample_points = np.load(sample_points_file)
                self.draw_initial_sample_points()

        if walls_file is not None:
            if os.path.exists(walls_file):
                self.new_walls = np.load(walls_file)
                self.draw_initial_walls()

        if add_points or delete_points:
            self.attach_handler()

        if add_walls:
            self.attach_wall_handler(correct_walls)

        if walls is not None:
            self.new_walls = walls
            self.draw_initial_walls()

    def attach_handler(self):
        def onclick(event):
            x = event.xdata
            y = event.ydata

            if self.delete_points:
                dist = np.linalg.norm(self.sample_points - np.array([x, y]), axis=1)
                min_idx = np.argmin(dist)
                min_dist = dist[min_idx]
                if min_dist < self.delete_th:
                    self.sample_points = np.delete(self.sample_points, min_idx, axis=0)
            else:
                new_point = np.array([x,y]).reshape(1,2)
                if self.sample_points is not None:
                    self.sample_points = np.concatenate((self.sample_points,
                                                   new_point), axis=0)
                else:
                    self.sample_points = new_point

                self.axis.scatter(x, y, s=50, color='r')

            self.fig.canvas.draw()

        def handle_close(event):
            np.save(self.sample_points_file, self.sample_points)

        self.fig.canvas.mpl_connect('button_press_event', onclick)
        self.fig.canvas.mpl_connect('close_event', handle_close)

    def attach_wall_handler(self, correct=True):
        def onclick(event):

            if self.nav_bar.mode != '':
                return

            x = event.xdata
            y = event.ydata

            if self.start_wall_point is None:
                self.start_wall_point = [x,y]
            else:
                s_x, s_y = self.start_wall_point[0], self.start_wall_point[1]
                self.start_wall_point = None

                if correct:
                    diff_x = abs(s_x - x)
                    diff_y = abs(s_y - y)
                    if diff_x < diff_y:
                        # vertical wall
                        x_mean = np.mean([s_x, x])
                        wall = np.array([x_mean, s_y, x_mean, y]).reshape(1,4)
                    else:
                        # horizontal wall
                        y_mean = np.mean([s_y, y])
                        wall = np.array([s_x, y_mean, x, y_mean]).reshape(1,4)
                else:
                    wall = np.array([s_x, s_y, x, y]).reshape(1,4)

                if self.new_walls is not None:
                    self.new_walls = np.concatenate((self.new_walls, wall), axis=0)
                else:
                    self.new_walls = wall

                self.axis.plot([wall[:,0], wall[:,2]], [wall[:,1], wall[:,3]], color='r')

            self.fig.canvas.draw()

        def handle_close(event):
            np.save(self.walls_file, self.new_walls)

        self.fig.canvas.mpl_connect('button_press_event', onclick)
        self.fig.canvas.mpl_connect('close_event', handle_close)

    def save_plot(self, bbox_inches=None):
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rc("savefig", dpi=200)

        pdf = PdfPages(self.filename)

        # self.fig.set_size_inches((19.79, 12.5), forward=False)
        pdf.savefig(bbox_inches=bbox_inches)
        pdf.close()

    def show_plot(self):
        plt.show()

    def set_title(self, title="title"):
        self.axis.set_title(title)

    def init_plot(self):
        fig, ax = plt.subplots(1, 1)

        self.axis = ax
        self.fig = fig

        self.nav_bar = fig.canvas.toolbar

    def draw_initial_sample_points(self):
        np.set_printoptions(suppress=True)
        print(np.round(self.sample_points, decimals=2))
        self.axis.scatter(self.sample_points[:,0], self.sample_points[:,1],
                          s=50, color='r')

    def draw_initial_walls(self, linewidth=2):

        if self.new_walls is not None:
            self.axis.plot([self.new_walls[:,0], self.new_walls[:,2]],
                           [self.new_walls[:,1], self.new_walls[:,3]],
                           color='red', linestyle="--", linewidth=linewidth)

    def draw_grid(self, color='grey', linewidth=.7):
        if self.axis is None:
            self.init_plot()
        # draw horizontal lines
        y_offset = 0
        dim_x = self.floorplan_dimensions[0]
        dim_y = self.floorplan_dimensions[1]
        grid_size = self.grid_size

        while y_offset < dim_y:
            self.axis.plot([0, dim_x], [y_offset, y_offset],
                    color=color, linewidth=linewidth)
            y_offset += grid_size

        # draw vertical lines
        x_offset = 0
        while x_offset < dim_x:
            self.axis.plot([x_offset, x_offset], [0, dim_y],
                    color=color, linewidth=linewidth)
            x_offset += grid_size

    def draw_lines(self, x_points, y_points, color="black", **kwargs):
        if self.axis is None:
            self.init_plot()
        self.axis.plot(x_points, y_points, color=color, **kwargs)

    def draw_background(self):
        if self.axis is None:
            self.init_plot()
        # bg image computations
        try:
            bg_image = plt.imread(self.floorplan_bg_image)
            self.bg_img = self.axis.imshow(bg_image, extent=[0, 0 + self.floorplan_dimensions[0],
                                                             0, 0 + self.floorplan_dimensions[1]],
                                           alpha=self.bg_alpha)
        except FileNotFoundError:
            print("No background image found")

    def draw_points(self, x_points, y_points, color='b', alpha=1, **kwargs):
        if self.axis is None:
            self.init_plot()
        # plot raw points
        return self.axis.scatter(x_points, y_points, color=color, alpha=alpha, **kwargs)

    def draw_polygons(self, polygons: List[Polygon], bbox=False, color='b', linewidth=1, linestyle="-", fill=False, alpha=None):
        if self.axis is None:
            self.init_plot()

        if color is None:
            pal = sns.color_palette("dark", n_colors=len(polygons))
        else:
            pal = [color] * len(polygons)

        for idx, polygon in enumerate(polygons):
            if bbox:
                min_x, min_y, max_x, max_y = polygon.bounds
                width = max_x - min_x
                height = max_y - min_y
                rect = patches.Rectangle((min_x, min_y), width, height, fill=fill, color=pal[idx], linewidth=linewidth, linestyle=linestyle, alpha=alpha)
                self.axis.add_patch(rect)
            else:
                x, y = polygon.exterior.xy
                self.axis.plot(x, y, color=pal[idx], linewidth=linewidth, linestyle=linestyle, alpha=alpha)

    def set_tick_frequency(self, frequency=1.0):
        self.axis.set_xticks(np.arange(0, self.floorplan_dimensions[0], frequency))
        self.axis.set_yticks(np.arange(0, self.floorplan_dimensions[1], frequency))


def init_floorplan(show_markers=False, annotate=False, floor=None, title=None, filename=None, axis=None, strict_cutoff=False, set_axis_labels=False, evaluation_path=False):
    sns.set_theme(style="white")

    if floor is None:
        floor = FLOOR

    base_path = EVALUATION_PATH if evaluation_path else DATA_PATH

    floorplan_dims = pd.read_csv(root + base_path + "floor_dimensions.csv").iloc[:, [1, 2]].to_numpy()

    if title is None:
        title = ""

    floor_plotter = FloorplanPlot(
        floorplan_dimensions=floorplan_dims[floor],
        grid_size=4,
        axis=axis,
        filename=filename,
        title=title,
        background_alpha=0.5,
        floorplan_bg_img=root + base_path + "floor_{}/floorplan.jpg".format(floor))

    if strict_cutoff:
        floor_plotter.axis.set_xlim(0, floorplan_dims[floor][0])
        floor_plotter.axis.set_ylim(0, floorplan_dims[floor][1])

    if show_markers:
        marker_path = root + base_path + "floor_{}".format(floor)
        marker_dict = get_marker_dict(marker_path)

        for k, v in marker_dict.items():
            floor_plotter.draw_points(v[0], v[1], marker=".", s=40, zorder=10, color=LANDMARK_COL)

            if annotate:
                floor_plotter.axis.annotate("%i" % k, (v[0], v[1]), zorder=10)

    floor_plotter.axis.grid(False)

    if set_axis_labels:
        floor_plotter.axis.set_xlabel("x [m]")
        floor_plotter.axis.set_xlabel("y [m]")

    return floor_plotter


if __name__ == "__main__":
    fp = init_floorplan(show_markers=True, floor=1, annotate=True)
    fp.show_plot()