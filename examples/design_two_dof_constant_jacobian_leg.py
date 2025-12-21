import sys
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import any_linkage.topology as topology
import any_linkage.dimensions as dimensions
import any_linkage.designer as designer


class TwoDoFConstantJacobianLegDesign(designer.Design):
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
                n_motors == 2 and
                n_links <= 5 and
                n_links_to_output == 2
            ):
                filtered_plans.append(plan)
        return filtered_plans

    def __init__(self, plan_index, seed=0):
        super().__init__(plan_index, seed=seed)
        self.plotter_bbox = (-200, -300, 400, 400)

        self.plan = TwoDoFConstantJacobianLegDesign.plans()[self.plan_index]
        self.g = topology.gen_graph(self.plan)
        self.c_empty = dimensions.gen_constraints(self.plan)
        self.origin_key = dimensions.origin_key
        self.output_key = dimensions.get_output_key(self.c_empty)

        self.n_designs = 1000

        self.cos_max = torch.tensor(0.8).to(self.device)
        self.output_clearance_min = torch.tensor(20).to(self.device)
        self.width_max = torch.tensor(100).to(self.device)

        self.weights = torch.tensor([
            1, 0.001, 1000, 1, 1,
        ]).to(self.device)

        self.p0 = {}
        for key in dimensions.get_point_keys(self.c_empty):
            p = torch.zeros([self.n_designs, 2]).to(self.device)
            p[:, 0].uniform_(-200, 200)
            p[:, 1].uniform_(-300, 100)
            p.requires_grad_(True)
            self.params.append(p)
            self.p0[key] = p

        grid = torch.meshgrid(
            torch.linspace(-1, 1, 5) - np.pi / 2,
            torch.linspace(200, 100, 5),
            indexing="ij",
        )
        polar = torch.stack([axis.flatten() for axis in grid]).T

        self.p_output_d = torch.zeros([polar.shape[0], 2]).to(self.device)
        self.p_output_d[:, 0] = polar[:, 1] * torch.cos(polar[:, 0])
        self.p_output_d[:, 1] = polar[:, 1] * torch.sin(polar[:, 0])

        jac = torch.zeros(self.n_designs, 2, 2).to(self.device)
        jac[:, 0, :].uniform_(-1, 1)
        jac[:, 1, :].uniform_(-100, 100)
        self.jac_scale = torch.tensor(
            [100, 1],
        ).expand(2, -1).T.to(self.device)
        self.jac_scaled = jac * self.jac_scale
        self.jac_scaled.requires_grad_(True)
        self.params.append(self.jac_scaled)

        self.polar = polar.to(self.device)

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

        jac_inv = torch.linalg.inv(self.jac_scaled / self.jac_scale)
        q = torch.matmul(
            jac_inv.unsqueeze(1), self.polar.unsqueeze(-1),
        ).squeeze(-1)
        q = q - torch.mean(q, dim=1, keepdim=True)

        p, cos_theta, cos_theta_p, cos_mu = dimensions.fk(
            q, self.p0, c,
        )
        if cos_theta is None:
            cos = torch.zeros(self.n_designs, device=self.device)
        else:
            cos = torch.stack([cos_theta, cos_theta_p, cos_mu], dim=2)
            cos = torch.amax(torch.abs(cos), dim=(1, 2, 3))

        loss_output_error = torch.mean(
            torch.sum(
                (p[self.output_key] - self.p_output_d)**2,
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

        loss_cos = torch.maximum(
            cos, self.cos_max,
        ) - self.cos_max

        p_output = p[self.output_key]
        rot_y = -p_output
        rot_y = rot_y / torch.linalg.norm(rot_y, dim=-1, keepdim=True)
        rot_x = torch.stack([rot_y[:, :, 1], -rot_y[:, :, 0]], dim=-1)
        rot = torch.stack([rot_x, rot_y], dim=-2)
        p_other = torch.stack(
            [v for k, v in p.items() if k != self.output_key],
            dim=2,
        )
        p_other = torch.matmul(
            rot.unsqueeze(2), p_other.unsqueeze(-1),
        ).squeeze(-1)
        p_output = torch.matmul(
            rot, p_output.unsqueeze(-1),
        ).squeeze(-1)
        output_clearance = (
            p_output[:, :, 1] -
            torch.amin(p_other[:, :, :, 1], dim=-1)
        )
        output_clearance = torch.amax(output_clearance, dim=-1)
        loss_output_clearance = torch.maximum(
            output_clearance, -self.output_clearance_min,
        ) - -self.output_clearance_min

        x_max = torch.amax(p_other[:, :, :, 0], dim=-1)
        x_min = torch.amin(p_other[:, :, :, 0], dim=-1)
        width = torch.amax(x_max - x_min, dim=-1)
        loss_width = torch.maximum(
            width, self.width_max,
        ) - self.width_max

        loss_itemized = torch.stack(
            [
                loss_output_error,
                loss_total_link_length,
                loss_cos,
                loss_output_clearance,
                loss_width,
            ],
            dim=1,
        )
        self.loss_weighted = self.weights * loss_itemized
        loss = torch.sum(self.weights * loss_itemized, dim=1)

        return loss, q, p, c

    def _on_plotted(self, d_index, q_index):
        p_output = self.p[self.output_key][d_index].detach().cpu().numpy()
        p_output_d = self.p_output_d.detach().cpu().numpy()

        plt.plot(p_output[:, 0], p_output[:, 1], '.b', lw=1)
        plt.plot(p_output_d[:, 0], p_output_d[:, 1], '.g', lw=1)

    def _on_design_changed(self, d_index, q_index):
        print(
            f"design: {d_index}, "
            f"l: {self.loss[d_index]:.4f}, "
            f"l_ae: {self.loss_weighted[d_index][0]:.4f}, "
            f"l_tll: {self.loss_weighted[d_index][1]:.4f}, "
            f"l_cos: {self.loss_weighted[d_index][2]:.4f}, "
            f"l_oc: {self.loss_weighted[d_index][3]:.4f}, "
            f"l_w: {self.loss_weighted[d_index][4]:.4f}"
        )
        jac = (
            self.jac_scaled[d_index] / self.jac_scale
        ).detach().cpu().numpy()
        q_std = torch.std(self.q[d_index], dim=0).detach().cpu().numpy()
        with np.printoptions(precision=4, suppress=True, floatmode="fixed"):
            print("jac: ")
            print(jac)
            print(f"q_std: {q_std}")


def main():
    if sys.argv[1] == "t":
        plan_index = int(sys.argv[2])
        design = TwoDoFConstantJacobianLegDesign(plan_index)
        design.eval()
        design.plot()
        plt.show()

    if sys.argv[1] == "o":
        plan_index = int(sys.argv[2])
        design = TwoDoFConstantJacobianLegDesign(plan_index)
        designer.optimize(design, id=plan_index)
        designer.save(design, "logs", name="two_dof_constant_jacobian_leg")

    if sys.argv[1] == "s":
        designer.sweep(
            TwoDoFConstantJacobianLegDesign,
            name="two_dof_constant_jacobian_leg",
            processes=2,
        )

    if sys.argv[1] == "p":
        path = sys.argv[2]
        design = designer.load(path)
        design.plot()
        plt.show()


if __name__ == "__main__":
    main()
