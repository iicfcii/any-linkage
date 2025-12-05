import sys
import torch
import networkx as nx
import matplotlib.pyplot as plt
import any_linkage.topology as topology
import any_linkage.dimensions as dimensions
import any_linkage.design as design
from any_linkage.plotter import Plotter


class OneDoFLegDesigner():
    def __init__(self, plan_index, seed=0, device="cuda"):
        self.plan_index = plan_index
        self.seed = seed
        torch.manual_seed(seed)
        self.device = device

        self.plan = OneDoFLegDesigner.plans()[self.plan_index]
        self.c_empty = dimensions.gen_constraints(self.plan)
        self.output_key = dimensions.get_output_key(self.c_empty)

        self.n_designs = 1000
        self.n_q_combs = 10

        self.cos_max = torch.tensor(0.8).to(self.device)

        self.weights = torch.tensor([1, 1000]).to(self.device)

        self.params = []
        self.p0 = {}
        for key in dimensions.get_point_keys(self.c_empty):
            if key == dimensions.origin_key:
                p = torch.zeros([self.n_designs, 2]).to(self.device)
            else:
                p = torch.zeros([self.n_designs, 2]).to(self.device)
                p[:, 0].uniform_(-100, 100)
                p[:, 1].uniform_(-100, 100)
                p.requires_grad_(True)
                self.params.append(p)
            self.p0[key] = p

        self.q = torch.linspace(
            -1, 1, self.n_q_combs,
        ).unsqueeze(-1).expand(self.n_designs, -1, -1).to(self.device)

        self.p_output_d = torch.zeros([self.n_q_combs, 2]).to(self.device)
        self.p_output_d[:, 1] = torch.linspace(
            -100, 0, self.n_q_combs,
        ).to(self.device)

    @staticmethod
    def plans():
        plans = topology.load()

        filtered_plans = []
        for plan in plans:
            g = topology.gen_graph(plan)
            n_motors = len([
                e
                for e in g.edges
                if g[e[0]][e[1]]["type"] == "m"
            ])
            n_links = len(g.nodes)
            n_links_to_output = nx.shortest_path_length(
                g, list(g.nodes)[0], list(g.nodes)[-1]
            )
            if (
                n_motors == 1 and
                n_links <= 6 and
                n_links_to_output >= 2
            ):
                filtered_plans.append(plan)
        return filtered_plans

    def eval(self):
        self.c = dimensions.populate(self.p0, self.c_empty)
        self.p, cos_theta, cos_theta_p, cos_mu = dimensions.fk(
            self.q, self.p0, self.c,
        )

        loss_output_error = torch.mean(
            torch.linalg.norm(
                self.p[self.output_key] - self.p_output_d,
                dim=-1,
            ), dim=-1,
        )

        cos = torch.cat([cos_theta, cos_theta_p, cos_mu], dim=1)
        cos_max = torch.amax(torch.abs(cos), dim=(1, 2))
        loss_cos_max = torch.maximum(cos_max, self.cos_max) - self.cos_max

        self.loss_itemized = torch.stack(
            [
                loss_output_error,
                loss_cos_max,
            ],
            dim=1,
        )
        self.loss_weighted = self.weights * self.loss_itemized
        self.loss_weighted_sum = torch.sum(
            self.weights * self.loss_itemized, dim=1,
        )

        return self.loss_weighted_sum

    def plot(self):
        indices = torch.argsort(self.loss_weighted_sum)
        self.q = self.q[indices]
        for k, v in self.p.items():
            self.p[k] = v[indices]
        self.c = [
            _c[:4] + [_c[4][indices], _c[5][indices]]
            for _c in self.c
        ]

        self.plotter = Plotter(
            self.q, self.p, self.c,
            (-200, -200, 400, 400),
            self.on_plotted,
            self.on_design_changed,
        )

    def on_plotted(self, d_index, q_index):
        p_output = self.p[self.output_key][d_index].detach().cpu().numpy()
        p_output_d = self.p_output_d.detach().cpu().numpy()
        plt.plot(p_output[:, 0], p_output[:, 1], 'b', lw=1)
        plt.plot(p_output_d[:, 0], p_output_d[:, 1], 'g', lw=1)

    def on_design_changed(self, d_index, q_index):
        print(
            f"design: {d_index}, "
            f"l: {self.loss_weighted_sum[d_index]:.4f}, "
            f"l_oe: {self.loss_weighted[d_index][0]:.4f}, "
            f"l_cos: {self.loss_weighted[d_index][1]:.4f}"
        )


def main():
    if sys.argv[1] == "t":
        plan_index = int(sys.argv[2])
        designer = OneDoFLegDesigner(plan_index)
        designer.eval()
        designer.plot()
        plt.show()

    if sys.argv[1] == "o":
        plan_index = int(sys.argv[2])
        designer = OneDoFLegDesigner(plan_index)
        design.optimize(designer.params, designer.eval, id=plan_index)
        design.save("logs", designer, name="one_dof_leg")

    if sys.argv[1] == "s":
        design.sweep(OneDoFLegDesigner, name="one_dof_leg", processes=2)

    if sys.argv[1] == "p":
        path = sys.argv[2]
        designer = design.load(path)
        designer.plot()
        plt.show()


if __name__ == "__main__":
    main()
