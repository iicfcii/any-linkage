import torch
import networkx as nx
import matplotlib.pyplot as plt
from any_linkage.topology import load, gen_graph
from any_linkage.dimensions import gen_constraints, get_point_keys, populate, fk, plot
from any_linkage.plotter import Plotter


def main():
    plans = load()

    filtered_plans = []
    for plan in plans:
        g = gen_graph(plan)
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
            n_motors == 2 and
            n_links == 7 and
            n_links_to_output == 2
        ):
            filtered_plans.append(plan)
    plan = filtered_plans[0]  # Two DoF seven-bar parallel topology

    torch.manual_seed(0)

    # There is a batch dimension for designs to calculate forward kinematics in parallel.
    n_designs = 2
    c = gen_constraints(plan)
    keys = get_point_keys(c)

    # Randomly sample the necessary number of points to form designs.
    # The eps value for avoiding division by zero error is set to 1e-6.
    # The default torch.float32 precision is not high, so please use units that result in larger values.
    p0 = {}
    for key in keys:
        p = torch.zeros([n_designs, 2])
        p[:, 0].uniform_(-100, 100)
        p[:, 1].uniform_(-100, 100)
        p0[key] = p

    # Specify the input ranges.
    grid = torch.meshgrid(
        torch.linspace(-1, 1, 10),
        torch.linspace(-1, 1, 10),
        indexing="ij",
    )
    q = torch.stack([axis.flatten() for axis in grid]).T
    # Accomodate the batch dimension for designs.
    # This allows differnt input ranges for different designs if needed.
    q = q.expand(n_designs, *q.shape)

    # Populate the constraints with values calculated from sampled designs.
    c = populate(p0, c)

    # Calculate forward kinematics.
    # The q combinations are also treated as a batch dimension.
    # The returned data has batch dimensions of n_designs x n_q_combs.
    p, cos_theta, cos_theta_p, cos_mu = fk(q, p0, c)

    # Random dimensions may result in kinematically infeasible designs,
    # which is indicated by the magnitude of cos_theta exceeding 1.
    # During optimization, constraints can be placed on it to avoid infeasible designs.
    # These values can be constrained to stay away from singularities too.
    cos_theta_max = torch.amax(torch.abs(cos_theta), dim=(-1, -2))
    cos_theta_p_max = torch.amax(torch.abs(cos_theta_p), dim=(-1, -2))
    cos_mu_max = torch.amax(torch.abs(cos_mu), dim=(-1, -2))
    print("cos_theta_max", cos_theta_max)
    print("cos_theta_p_max", cos_theta_p_max)
    print("cos_mu_max", cos_mu_max)

    plotter = Plotter(q, p, c, (-200, -200, 400, 400))
    plt.show()


if __name__ == "__main__":
    main()
