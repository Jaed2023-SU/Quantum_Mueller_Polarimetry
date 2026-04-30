import numpy as np
import matplotlib.pyplot as plt


class PoincarePlot:
    def __init__(
        self,
        show_reference_points=True,
        show_reference_labels=True,
        trajectory_marker='.',
        elev=20,
        azim=35,
        sphere_alpha=0.1,
        figsize=(9, 9),
    ):
        self.show_reference_points = show_reference_points
        self.show_reference_labels = show_reference_labels
        self.trajectory_marker = trajectory_marker
        self.elev = elev
        self.azim = azim
        self.sphere_alpha = sphere_alpha
        self.figsize = figsize

        self.ref_points = {
            "H": (1, 0, 0),
            "V": (-1, 0, 0),
            "D": (0, 1, 0),
            "A": (0, -1, 0),
            "R": (0, 0, 1),
            "L": (0, 0, -1),
        }

    def _validate_and_normalize_stokes(self, stokes_vectors):
        if stokes_vectors is None:
            return None

        stokes_vectors = np.asarray(stokes_vectors, dtype=float)

        if stokes_vectors.ndim == 1:
            if stokes_vectors.shape[0] != 3:
                raise ValueError("If 1D input is used, it must have shape (3,).")
            stokes_vectors = stokes_vectors.reshape(1, 3)

        if stokes_vectors.ndim != 2 or stokes_vectors.shape[1] != 3:
            raise ValueError("stokes_vectors must have shape (N,3) or (3,)")

        norms = np.linalg.norm(stokes_vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Zero-norm Stokes vector is not allowed.")

        stokes_norm = stokes_vectors / norms
        return stokes_norm

    def _draw_sphere(self, ax):
        u = np.linspace(0, 2 * np.pi, 120)
        v = np.linspace(0, np.pi, 60)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))

        ax.plot_surface(
            x,
            y,
            z,
            edgecolor='gray',
            linewidth=0.05,
            alpha=self.sphere_alpha,
        )

    def _draw_center_axes(self, ax):
        axis_len = 1.05
        ax.plot([-axis_len, axis_len], [0, 0], [0, 0], color='k', linewidth=0.5)
        ax.plot([0, 0], [-axis_len, axis_len], [0, 0], color='k', linewidth=0.5)
        ax.plot([0, 0], [0, 0], [-axis_len, axis_len], color='k', linewidth=0.5)

    def _draw_axis_end_labels(self, ax):
        label_offset = 1.4
        ax.text(label_offset, 0, 0, "H", fontsize=14, ha='center', va='center')
        ax.text(0, label_offset, 0, "D", fontsize=14, ha='center', va='center')
        ax.text(0, 0, label_offset, "R", fontsize=14, ha='center', va='center')
        ax.text(-label_offset, 0, 0, "V", fontsize=14, ha='center', va='center')
        ax.text(0, -label_offset, 0, "A", fontsize=14, ha='center', va='center')
        ax.text(0, 0, -label_offset, "L", fontsize=14, ha='center', va='center')

    def _draw_reference_points(self, ax):
        if not self.show_reference_points:
            return

        for label, (s1, s2, s3) in self.ref_points.items():
            ax.scatter([s1], [s2], [s3], color='k', marker='^', s=50, depthshade=False)

            if self.show_reference_labels:
                scale = 1.08
                ax.text(
                    scale * s1,
                    scale * s2,
                    scale * s3,
                    label,
                    fontsize=10,
                    ha='center',
                    va='center',
                )

    def _draw_great_circles(self, ax):
        phi = np.linspace(0, 2 * np.pi, 400)
        gx = np.cos(phi)
        gy = np.sin(phi)
        gz = np.zeros_like(phi)

        ax.plot(gx, gy, gz, color='b', linestyle='--', linewidth=0.5)
        ax.plot(gz, gy, gx, color='b', linestyle='--', linewidth=0.5)
        ax.plot(gx, gz, gy, color='b', linestyle='--', linewidth=0.5)

    def _draw_trajectory(self, ax, stokes_vectors):
        if stokes_vectors is None:
            return

        stokes_norm = self._validate_and_normalize_stokes(stokes_vectors)

        s1 = stokes_norm[:, 0]
        s2 = stokes_norm[:, 1]
        s3 = stokes_norm[:, 2]

        ax.plot(s1, s2, s3, linewidth=1.8)
        ax.scatter(
            s1,
            s2,
            s3,
            marker=self.trajectory_marker,
            s=35,
            depthshade=False,
        )

        ax.scatter(
            [s1[0]],
            [s2[0]],
            [s3[0]],
            color='k',
            marker='s',
            s=70,
            depthshade=False,
        )
        ax.scatter(
            [s1[-1]],
            [s2[-1]],
            [s3[-1]],
            color='k',
            marker='*',
            s=130,
            depthshade=False,
        )

    def _style_axes(self, ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.grid(False)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.xaxis.pane.set_edgecolor((1, 1, 1, 1))
        ax.yaxis.pane.set_edgecolor((1, 1, 1, 1))
        ax.zaxis.pane.set_edgecolor((1, 1, 1, 0))

        ax.set_axis_off()

        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis._axinfo["grid"]["linewidth"] = 0
            axis._axinfo["tick"]["inward_factor"] = 0
            axis._axinfo["tick"]["outward_factor"] = 0
            axis._axinfo["axisline"]["linewidth"] = 0

        lim = 1.28
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_zlim([-lim, lim])
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=self.elev, azim=self.azim)

    def plot_poincare(self, stokes_vectors=None, show=True):
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')

        self._draw_sphere(ax)
        self._draw_center_axes(ax)
        self._draw_axis_end_labels(ax)
        self._draw_reference_points(ax)
        self._draw_great_circles(ax)
        self._draw_trajectory(ax, stokes_vectors)
        self._style_axes(ax)

        plt.tight_layout()

        if show:
            plt.show()

        return fig, ax