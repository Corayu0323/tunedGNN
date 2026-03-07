import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch_scatter import scatter
from torch_geometric.loader import NeighborLoader
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from models_pyg import GNN_PyG

device = None
dataset = "ogbn-proteins"
n_node_feats, n_edge_feats, n_classes = 0, 8, 112


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(dataset):
    dataset_obj = PygNodePropPredDataset(name=dataset, root='data/ogb')
    evaluator = Evaluator(name=dataset)

    splitted_idx = dataset_obj.get_idx_split()
    train_idx = splitted_idx["train"]
    val_idx = splitted_idx["valid"]
    test_idx = splitted_idx["test"]
    data = dataset_obj[0]

    return data, train_idx, val_idx, test_idx, evaluator


def preprocess(data, train_idx):
    global n_node_feats

    # Aggregate edge features to produce node features via sum of incoming edge features.
    # This replicates the DGL update_all(copy_e, sum) operation.
    x = scatter(data.edge_attr, data.edge_index[1], dim=0,
                dim_size=data.num_nodes, reduce='sum')
    data.x = x
    n_node_feats = data.x.shape[-1]

    # Only labels in the training set are used as features; others are filled with zeros.
    data.train_labels_onehot = torch.zeros(data.num_nodes, n_classes)
    data.train_labels_onehot[train_idx, data.y[train_idx, 0].long()] = 1

    return data


def gen_model(args):
    if args.use_labels:
        n_node_feats_ = n_node_feats + n_classes
    else:
        n_node_feats_ = n_node_feats

    model = GNN_PyG(
        n_node_feats_,
        n_edge_feats,
        n_classes,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_hidden=args.n_hidden,
        edge_emb=16,
        activation=F.relu,
        dropout=args.dropout,
        input_drop=args.input_drop,
        attn_drop=args.attn_drop,
        edge_drop=args.edge_drop,
        use_attn_dst=not args.no_attn_dst,
        mpnn=args.mpnn,
        jk=args.jk,
    )

    return model


def add_labels(x, train_labels_onehot, idx):
    """Concatenate one-hot training labels to node features for the given local indices."""
    labels_onehot = torch.zeros([x.shape[0], n_classes], device=x.device)
    labels_onehot[idx] = train_labels_onehot[idx].to(x.device)
    return torch.cat([x, labels_onehot], dim=-1)


def train(args, model, dataloader, criterion, optimizer):
    model.train()

    loss_sum, total = 0, 0

    for batch in dataloader:
        batch = batch.to(device)
        batch_size = batch.batch_size

        if args.use_labels:
            # Add training labels for non-seed nodes (nodes beyond the first batch_size)
            non_seed_idx = torch.arange(batch_size, batch.x.shape[0], device=device)
            x = add_labels(batch.x, batch.train_labels_onehot, non_seed_idx)
        else:
            x = batch.x

        pred = model(x, batch.edge_index, batch.edge_attr)
        loss = criterion(pred[:batch_size], batch.y[:batch_size].float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * batch_size
        total += batch_size

    return loss_sum / total


@torch.no_grad()
def evaluate(args, model, dataloader, labels, train_idx, val_idx, test_idx, criterion, evaluator):
    model.eval()

    preds = torch.zeros(labels.shape, device=device)

    # Due to memory constraints, use sampling for inference; average over eval_times passes.
    eval_times = 1

    for _ in range(eval_times):
        for batch in dataloader:
            batch = batch.to(device)
            batch_size = batch.batch_size

            if args.use_labels:
                # During evaluation, add labels for all nodes in the subgraph
                all_idx = torch.arange(batch.x.shape[0], device=device)
                x = add_labels(batch.x, batch.train_labels_onehot, all_idx)
            else:
                x = batch.x

            pred = model(x, batch.edge_index, batch.edge_attr)
            # batch.n_id maps local node indices to global node indices
            preds[batch.n_id[:batch_size]] += pred[:batch_size]

    preds /= eval_times

    train_loss = criterion(preds[train_idx], labels[train_idx].float()).item()
    val_loss = criterion(preds[val_idx], labels[val_idx].float()).item()
    test_loss = criterion(preds[test_idx], labels[test_idx].float()).item()

    return (
        evaluator(preds[train_idx], labels[train_idx]),
        evaluator(preds[val_idx], labels[val_idx]),
        evaluator(preds[test_idx], labels[test_idx]),
        train_loss,
        val_loss,
        test_loss,
        preds,
    )


def run(args, data, labels, train_idx, val_idx, test_idx, evaluator, n_running):
    evaluator_wrapper = lambda pred, labels: evaluator.eval({"y_pred": pred, "y_true": labels})["rocauc"]

    train_batch_size = (len(train_idx) + 9) // 10

    train_loader = NeighborLoader(
        data,
        num_neighbors=[32] * args.n_layers,
        batch_size=train_batch_size,
        input_nodes=train_idx.cpu(),
        shuffle=True,
        num_workers=10,
    )

    eval_loader = NeighborLoader(
        data,
        num_neighbors=[100] * args.n_layers,
        batch_size=65536,
        input_nodes=torch.cat([train_idx.cpu(), val_idx.cpu(), test_idx.cpu()]),
        shuffle=False,
        num_workers=10,
    )

    criterion = nn.BCEWithLogitsLoss()

    model = gen_model(args).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.75, patience=50, verbose=True
    )

    total_time = 0
    val_score, best_val_score, final_test_score = 0, 0, 0

    train_scores, val_scores, test_scores = [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []
    final_pred = None

    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()

        loss = train(args, model, train_loader, criterion, optimizer)

        toc = time.time()
        total_time += toc - tic

        if epoch == args.n_epochs or epoch % args.eval_every == 0 or epoch % args.log_every == 0:
            train_score, val_score, test_score, train_loss, val_loss, test_loss, pred = evaluate(
                args, model, eval_loader, labels, train_idx, val_idx, test_idx, criterion, evaluator_wrapper
            )

            if val_score > best_val_score:
                best_val_score = val_score
                final_test_score = test_score
                final_pred = pred

            if epoch % args.log_every == 0:
                print(
                    f'Epoch: {epoch:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * train_score:.2f}%, '
                    f'Valid: {100 * val_score:.2f}%, '
                    f'Test: {100 * test_score:.2f}%, '
                    f'Best Valid: {100 * best_val_score:.2f}%, '
                    f'Best Test: {100 * final_test_score:.2f}%'
                )

            for l, e in zip(
                [train_scores, val_scores, test_scores, losses, train_losses, val_losses, test_losses],
                [train_score, val_score, test_score, loss, train_loss, val_loss, test_loss],
            ):
                l.append(e)

        lr_scheduler.step(val_score)

    if args.save_pred:
        os.makedirs("./output", exist_ok=True)
        torch.save(F.softmax(final_pred, dim=1), f"./output/{n_running}.pt")

    return best_val_score, final_test_score


def count_parameters(args):
    model = gen_model(args)
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


def main():
    global device

    argparser = argparse.ArgumentParser(
        "GNN implementation on ogbn-proteins (PyG)", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides '--gpu'.")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    argparser.add_argument("--seed", type=int, default=0, help="random seed")
    argparser.add_argument("--n-runs", type=int, default=1, help="running times")
    argparser.add_argument("--n-epochs", type=int, default=1000, help="number of epochs")
    argparser.add_argument("--mpnn", type=str, default='gat')
    argparser.add_argument(
        "--use-labels", action="store_true", help="Use labels in the training set as input features."
    )
    argparser.add_argument("--no-attn-dst", action="store_true", help="Don't use attn_dst.")
    argparser.add_argument("--n-heads", type=int, default=6, help="number of heads")
    argparser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    argparser.add_argument("--n-layers", type=int, default=6, help="number of layers")
    argparser.add_argument("--n-hidden", type=int, default=80, help="number of hidden units")
    argparser.add_argument("--dropout", type=float, default=0.25, help="dropout rate")
    argparser.add_argument("--input-drop", type=float, default=0.1, help="input drop rate")
    argparser.add_argument("--attn-drop", type=float, default=0.0, help="attention dropout rate")
    argparser.add_argument("--edge-drop", type=float, default=0.1, help="edge drop rate")
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    argparser.add_argument("--eval-every", type=int, default=5, help="evaluate every EVAL_EVERY epochs")
    argparser.add_argument("--log-every", type=int, default=5, help="log every LOG_EVERY epochs")
    argparser.add_argument("--plot", action="store_true", help="plot learning curves")
    argparser.add_argument("--save-pred", action="store_true", help="save final predictions")
    argparser.add_argument("--jk", action="store_true")
    args = argparser.parse_args()

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    # load data & preprocess
    print("Loading data")
    data, train_idx, val_idx, test_idx, evaluator = load_data(dataset)
    print("Preprocessing")
    data = preprocess(data, train_idx)

    labels = data.y
    labels, train_idx, val_idx, test_idx = map(lambda x: x.to(device), (labels, train_idx, val_idx, test_idx))

    # run
    val_scores, test_scores = [], []

    for i in range(args.n_runs):
        print("Running", i)
        seed(args.seed + i)
        val_score, test_score = run(args, data, labels, train_idx, val_idx, test_idx, evaluator, i + 1)
        val_scores.append(val_score)
        test_scores.append(test_score)

    print(args)


if __name__ == "__main__":
    main()
