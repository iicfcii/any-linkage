import os
import time
import multiprocessing as mp
import torch
import torch.optim as optim
import numpy as np


def optimize(
    params, eval,
    lr=1, lr_schedule="cosine",
    n_steps=10000,
    log_interval=1000,
    id=None,
):
    optimizer = optim.Adam(params, lr=lr)

    if os.name != "nt":
        torch.compiler.reset()
        eval = torch.compile(eval)
    else:
        # Not using torch.compile on Windows because it is not well supported.
        # torch.compile can significantly speed up the optimization.
        pass

    t0 = time.time()
    for i in range(n_steps + 1):
        loss_weighted_sum = eval()
        loss_mean = torch.mean(loss_weighted_sum)

        if i > 0 and i % log_interval == 0:
            sps = i / (time.time() - t0)
            loss_min = torch.amin(loss_weighted_sum.detach())
            print(
                f"id: {id}, "
                f"iter: {i}, "
                f"sps: {sps:.2f}, "
                f"loss_mean: {loss_mean:.2f}, "
                f"loss_min: {loss_min:.2f}"
            )

        if i < n_steps:
            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()
            if lr_schedule == "linear":
                optimizer.param_groups[0]["lr"] = (1 - i / n_steps) * lr
            elif lr_schedule == "cosine":
                optimizer.param_groups[0]["lr"] = (
                    np.cos(i / n_steps * np.pi) + 1
                ) / 2 * lr
            else:
                pass


def save(path, designer, name="designer"):
    timestamp = int(time.time())
    loss_min = torch.amin(designer.loss_weighted_sum.detach()).item()
    data = {
        "timestamp": timestamp,
        "designer": designer,
    }
    name = f"{timestamp}_{name}_{designer.plan_index}_{loss_min:.4f}.pt"
    path = os.path.join(path, name)
    torch.save(data, path)


def load(path):
    return torch.load(path, weights_only=False)["designer"]


def _sweep_fun(Designer, plan_index, name, optimize_kwargs):
    designer = Designer(plan_index)
    optimize(designer.params, designer.eval, id=plan_index, **optimize_kwargs)
    save("logs", designer, name)


def sweep(Designer, processes=1, name="designer", optimize_kwargs={}):
    if processes > 1:
        num_plans = len(Designer.plans())
        args = [
            (Designer, i, name, optimize_kwargs)
            for i in range(num_plans)
        ]
        with mp.Pool(2) as p:
            p.starmap(_sweep_fun, args, chunksize=1)
    else:
        for plan_index in range(len(Designer.plans())):
            _sweep_fun(Designer, plan_index, name, optimize_kwargs)
