import sys
import torch
import networkx as nx
import matplotlib.pyplot as plt
import any_linkage.topology as topology
import any_linkage.dimensions as dimensions
import any_linkage.designer as designer


class OneDoFLinearLegDesign(designer.Design):
    def plans():
        plans = topology.load()

        filtered_plans = []
        for plan in plans:
            g = topology.gen_graph(plan)
            n_motors = len([
                e for e in g.edges
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
        self.plotter_bbox = (-200, -300, 400, 400)

        self.plan = OneDoFLinearLegDesign.plans()[self.plan_index]
        self.c_empty = dimensions.gen_constraints(self.plan)
        self.origin_key = dimensions.origin_key
        self.output_key = dimensions.get_output_key(self.c_empty)

        self.n_designs = 1000

        self.cos_max = torch.tensor(0.8).to(self.device)
        self.output_clearance_min = torch.tensor(20).to(self.device)
        self.joint_x_max = torch.tensor(50).to(self.device)

        self.weights = torch.tensor([1, 0.001, 1000, 1, 1]).to(self.device)

        self.p0 = {}
        for key in dimensions.get_point_keys(self.c_empty):
            p = torch.zeros([self.n_designs, 2]).to(self.device)
            p[:, 0].uniform_(-200, 200)
            p[:, 1].uniform_(-300, 100)
            p.requires_grad_(True)
            self.params.append(p)
            self.p0[key] = p

        self.q = torch.linspace(-1, 1, 9).unsqueeze(-1)

        jac = 50
        self.p_output_d = torch.zeros([self.q.shape[0], 2]).to(self.device)
        self.p_output_d[:, 1] = self.q.squeeze(-1) * jac - 150

        self.q = self.q.expand(self.n_designs, *self.q.shape).to(self.device)

        self.g = topology.gen_graph(self.plan)
        self.points_of_links = []
        for n in self.g.nodes:
            points = list(self.g.edges(n))
            if self.g.nodes[n]["type"] == "g":
                points.append(self.origin_key)
            if self.g.nodes[n]["type"] == "o":
                points.append(self.output_key)
            points = [tuple(sorted(list(point))) for point in points]
            self.points_of_links.append(points)

    def _eval(self):
        c = dimensions.populate(self.p0, self.c_empty)
        p, cos_theta, cos_theta_p, cos_mu = dimensions.fk(
            self.q, self.p0, c,
        )

        loss_output_error = torch.mean(
            torch.linalg.norm(
                p[self.output_key] - self.p_output_d,
                dim=-1,
            ),
            dim=-1,
        )

        centroid_link_length = []
        for points_of_link in self.points_of_links:
            _p = torch.stack(
                [self.p0[edge] for edge in points_of_link],
                dim=1,
            )
            # sum of distances to centroid
            centroid_link_length.append(torch.sum(
                torch.linalg.norm(
                    _p - torch.mean(_p, dim=1, keepdim=True),
                    dim=-1,
                ),
                dim=1,
            ))
        centroid_link_length = torch.stack(centroid_link_length, dim=0).T
        total_link_length = torch.sum(centroid_link_length, dim=1)
        loss_total_link_length = total_link_length

        cos = torch.cat([cos_theta, cos_theta_p, cos_mu], dim=1)
        cos = torch.amax(torch.abs(cos), dim=(1, 2))
        loss_cos_max = torch.maximum(
            cos, self.cos_max,
        ) - self.cos_max

        p_other = torch.stack(
            [v for k, v in p.items() if k != self.output_key],
            dim=2,
        )
        p_output = p[self.output_key]
        output_clearance = (
            p_output[:, :, 1] - torch.amin(p_other[:, :, :, 1], dim=-1)
        )
        output_clearance = torch.amax(output_clearance, dim=-1)
        loss_output_clearance = torch.maximum(
            output_clearance, -self.output_clearance_min,
        ) - -self.output_clearance_min

        p_all = torch.stack([v for k, v in p.items()], dim=2)
        joint_x = torch.amax(torch.abs(p_all[:, :, :, 0]), dim=(1, 2))
        loss_joint_x = torch.maximum(
            joint_x, self.joint_x_max,
        ) - self.joint_x_max

        loss_itemized = torch.stack(
            [
                loss_output_error,
                loss_total_link_length,
                loss_cos_max,
                loss_output_clearance,
                loss_joint_x,
            ],
            dim=1,
        )
        loss_weighted = self.weights * loss_itemized
        loss = torch.sum(self.weights * loss_itemized, dim=1)

        return loss, loss_weighted, p, c

    def _on_plotted(self, d_index, q_index):
        p_output = self.p[self.output_key][d_index].detach().cpu().numpy()
        p_output_d = self.p_output_d.detach().cpu().numpy()
        plt.plot(p_output[:, 0], p_output[:, 1], '.-b', lw=1)
        plt.plot(p_output_d[:, 0], p_output_d[:, 1], '.-g', lw=1)

    def _on_design_changed(self, d_index, q_index):
        print(
            f"design: {d_index}, "
            f"l: {self.loss[d_index]:.4f}, "
            f"l_oe: {self.loss_weighted[d_index][0]:.4f}, "
            f"l_tll: {self.loss_weighted[d_index][1]:.4f}, "
            f"l_cos: {self.loss_weighted[d_index][2]:.4f}, "
            f"l_oc: {self.loss_weighted[d_index][3]:.4f}, "
            f"l_jx: {self.loss_weighted[d_index][4]:.4f}"
        )


def main():
    if sys.argv[1] == "t":
        plan_index = int(sys.argv[2])
        design = OneDoFLinearLegDesign(plan_index)
        design.eval()
        design.plot()
        plt.show()

    if sys.argv[1] == "o":
        plan_index = int(sys.argv[2])
        design = OneDoFLinearLegDesign(plan_index)
        designer.optimize(design, id=plan_index)
        designer.save(design, "logs", name="one_dof_linear_leg")

    if sys.argv[1] == "s":
        designer.sweep(
            OneDoFLinearLegDesign,
            name="one_dof_linear_leg",
            processes=2,
        )

    if sys.argv[1] == "p":
        path = sys.argv[2]
        design = designer.load(path)
        design.plot()
        plt.show()


if __name__ == "__main__":
    main()
