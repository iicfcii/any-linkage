import os
from any_linkage.topology import load, plot, gen_graph
import networkx as nx
import matplotlib.pyplot as plt


def main():
    plans = load()

    filtered_plans = []
    for plan in plans:
        g = gen_graph(plan)
        n_motors = len([
            e for e in g.edges
            if g[e[0]][e[1]]["type"] == "m"
        ])
        n_links = len(g.nodes)
        n_links_to_output = nx.shortest_path_length(
            g, list(g.nodes)[0], list(g.nodes)[-1]
        )
        n_ground_joints = len(g.edges(0))
        n_ground_motors = len([
            e for e in g.edges(0)
            if g[e[0]][e[1]]["type"] == "m"
        ])
        if (
            n_motors <= 2 and
            n_links <= 5 and
            n_links_to_output >= 2 and
            n_ground_joints > 0 and
            n_ground_motors > 0
        ):
            filtered_plans.append(plan)

    plt.figure(figsize=(10, 4))
    plt.subplots_adjust(
        left=0, right=1, top=1, bottom=0,
        wspace=0, hspace=0,
    )
    for i, plan in enumerate(filtered_plans):
        if i >= 10:
            print(f"Too many topologies ({len(filtered_plans)}) to plot ")
            break
        plt.subplot(2, 5, i + 1)
        plot(plan)
    plt.show()


if __name__ == "__main__":
    main()
