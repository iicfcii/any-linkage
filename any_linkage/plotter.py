import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from any_linkage.dimensions import plot


class Plotter:
    def __init__(
        self,
        q, p, c, bbox,
        on_plotted=None,
        on_design_changed=None,
    ):
        self.q = q.detach().cpu().numpy()
        self.p = {}
        for k, v in p.items():
            self.p[k] = v.detach().cpu().numpy()
        self.c = [
            _c[:4] + [_c[4].detach().cpu().numpy(), _c[5].detach().cpu().numpy()]
            for _c in c
        ]

        self.bbox = bbox
        self.on_plotted = on_plotted
        self.on_design_changed = on_design_changed

        self.n_designs = self.q.shape[0]
        self.n_qs = self.q.shape[-1]

        self.d_index = 0
        self.q_index = 0
        self.label_dimensions = False

        self.fig_ctrl, self.axes_ctrl = plt.subplots(
            self.n_qs + 2, 1,
            num="ctrl", figsize=(8, 8),
        )
        self.fig_ctrl.subplots_adjust(
            left=0.2, right=0.8, top=0.9, bottom=0.1,
        )

        self.design_slider = Slider(
            self.axes_ctrl[-2],
            f"d",
            valmin=0,
            valmax=self.n_designs - 1,
            valinit=0,
            valstep=list(range(self.n_designs)),
            orientation="horizontal",
            initcolor="none",
        )
        self.design_slider.on_changed(self.on_design_slider_changed)

        self.label_slider = Slider(
            self.axes_ctrl[-1],
            f"l",
            valmin=0,
            valmax=1,
            valinit=0,
            valstep=[0, 1],
            orientation="horizontal",
            initcolor="none",
        )
        self.label_slider.on_changed(self.on_label_slider_changed)

        self.fig, self.ax = plt.subplots(num="design", figsize=(8, 8))
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        self.on_design_slider_changed(self.d_index)

    def on_design_slider_changed(self, val):
        self.d_index = val
        steps = []
        for i in range(self.n_qs):
            steps.append(np.unique(self.q[self.d_index, :, i]))
        self.q_sliders = []
        for i in range(self.n_qs):
            plt.sca(self.axes_ctrl[i])
            plt.cla()
            slider = Slider(
                self.axes_ctrl[i],
                f"q{i}",
                valmin=np.amin(steps[i]),
                valmax=np.amax(steps[i]),
                valinit=np.amin(steps[i]),
                valstep=steps[i],
                valfmt="%.2f",
                orientation="horizontal",
                initcolor="none",
            )
            slider.on_changed(self.on_q_slider_changed)
            self.q_sliders.append(slider)
        self.draw()

        if self.on_design_changed is not None:
            self.on_design_changed(self.d_index, self.q_index)

    def on_label_slider_changed(self, val):
        if val == 0:
            self.label_dimensions = False
        else:
            self.label_dimensions = True
        self.draw()

    def on_q_slider_changed(self, val):
        qi = np.array([slider.val for slider in self.q_sliders])
        self.q_index = np.argmin(np.linalg.norm(
            self.q[self.d_index] - qi, axis=-1,
        ))
        self.draw()

    def draw(self):
        plt.sca(self.ax)
        plt.cla()
        plot(
            self.p, self.c,
            self.d_index, self.q_index,
            label_dimensions=self.label_dimensions,
        )
        if self.on_plotted is not None:
            self.on_plotted(self.d_index, self.q_index)
        plt.xlim(self.bbox[0], self.bbox[0] + self.bbox[2])
        plt.ylim(self.bbox[1], self.bbox[1] + self.bbox[3])
        plt.draw()
