import sys
import torch
import networkx as nx
import matplotlib.pyplot as plt
import any_linkage.topology as topology
import any_linkage.dimensions as dimensions
import any_linkage.designer as designer


class OneDoFLegDesign(designer.Design):
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

    def __init__(self, plan_index, seed=0, device="cuda"):
        super().__init__(plan_index, seed=seed, device=device)
        self.plotter_bbox = (-200, -200, 400, 400)

        self.plan = OneDoFLegDesign.plans()[self.plan_index]
        self.c_empty = dimensions.gen_constraints(self.plan)
        self.output_key = dimensions.get_output_key(self.c_empty)

        self.n_designs = 1000
        self.n_q_combs = 10

        self.cos_max = torch.tensor(0.8).to(self.device)

        self.weights = torch.tensor([1, 1000]).to(self.device)

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

    def _eval(self):
        c = dimensions.populate(self.p0, self.c_empty)
        p, cos_theta, cos_theta_p, cos_mu = dimensions.fk(
            self.q, self.p0, c,
        )

        loss_output_error = torch.mean(
            torch.linalg.norm(
                p[self.output_key] - self.p_output_d,
                dim=-1,
            ), dim=-1,
        )

        cos = torch.cat([cos_theta, cos_theta_p, cos_mu], dim=1)
        cos_max = torch.amax(torch.abs(cos), dim=(1, 2))
        loss_cos_max = torch.maximum(cos_max, self.cos_max) - self.cos_max

        loss_itemized = torch.stack(
            [
                loss_output_error,
                loss_cos_max,
            ],
            dim=1,
        )
        self.loss_weighted = self.weights * loss_itemized
        loss = torch.sum(self.weights * loss_itemized, dim=1)

        return loss, p, c

    def _on_plotted(self, d_index, q_index):
        p_output = self.p[self.output_key][d_index].detach().cpu().numpy()
        p_output_d = self.p_output_d.detach().cpu().numpy()
        plt.plot(p_output[:, 0], p_output[:, 1], 'b', lw=1)
        plt.plot(p_output_d[:, 0], p_output_d[:, 1], 'g', lw=1)

    def _on_design_changed(self, d_index, q_index):
        print(
            f"design: {d_index}, "
            f"l: {self.loss[d_index]:.4f}, "
            f"l_oe: {self.loss_weighted[d_index][0]:.4f}, "
            f"l_cos: {self.loss_weighted[d_index][1]:.4f}"
        )


def main():
    if sys.argv[1] == "t":
        plan_index = int(sys.argv[2])
        design = OneDoFLegDesign(plan_index)
        design.eval()
        design.plot()
        plt.show()

    if sys.argv[1] == "o":
        plan_index = int(sys.argv[2])
        design = OneDoFLegDesign(plan_index)
        designer.optimize(design, id=plan_index)
        designer.save(design, "logs", name="one_dof_leg")

    if sys.argv[1] == "s":
        designer.sweep(OneDoFLegDesign, name="one_dof_leg", processes=2)

    if sys.argv[1] == "p":
        path = sys.argv[2]
        design = designer.load(path)
        design.plot()
        plt.show()


if __name__ == "__main__":
    main()
