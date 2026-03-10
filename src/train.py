import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import NeighborLoader

from .utils import add_labels, gen_model


def train_epoch(model, dataloader, criterion, optimizer, device,
                use_labels=False, n_classes=112):
    model.train()
    loss_sum, total = 0, 0

    for batch in dataloader:
        batch      = batch.to(device)
        batch_size = batch.batch_size

        if use_labels:
            non_seed_idx = torch.arange(batch_size, batch.x.shape[0], device=device)
            x = add_labels(batch.x, batch.train_labels_onehot, non_seed_idx, n_classes, device)
        else:
            x = batch.x

        pred = model(x, batch.edge_index, batch.edge_attr)
        loss = criterion(pred[:batch_size], batch.y[:batch_size].float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * batch_size
        total    += batch_size

    return loss_sum / total


@torch.no_grad()
def evaluate(model, dataloader, labels, train_idx, val_idx, test_idx,
             criterion, evaluator, device, use_labels=False, n_classes=112):
    model.eval()
    preds      = torch.zeros(labels.shape, device=device)
    eval_times = 1

    for _ in range(eval_times):
        for batch in dataloader:
            batch      = batch.to(device)
            batch_size = batch.batch_size

            if use_labels:
                all_idx = torch.arange(batch.x.shape[0], device=device)
                x = add_labels(batch.x, batch.train_labels_onehot, all_idx, n_classes, device)
            else:
                x = batch.x

            pred = model(x, batch.edge_index, batch.edge_attr)
            preds[batch.n_id[:batch_size]] += pred[:batch_size]

    preds /= eval_times

    train_loss = criterion(preds[train_idx], labels[train_idx].float()).item()
    val_loss   = criterion(preds[val_idx],   labels[val_idx].float()).item()
    test_loss  = criterion(preds[test_idx],  labels[test_idx].float()).item()

    return (
        evaluator(preds[train_idx], labels[train_idx]),
        evaluator(preds[val_idx],   labels[val_idx]),
        evaluator(preds[test_idx],  labels[test_idx]),
        train_loss, val_loss, test_loss,
        preds,
    )


def run(data, labels, train_idx, val_idx, test_idx, evaluator, n_running,
        gen_model_fn, device, n_layers, lr, weight_decay, n_epochs,
        eval_every, log_every, save_pred, use_labels=False, n_classes=112):
    evaluator_wrapper = lambda pred, lbls: evaluator.eval(
        {'y_pred': pred, 'y_true': lbls}
    )['rocauc']

    train_batch_size = (len(train_idx) + 9) // 10

    train_loader = NeighborLoader(
        data,
        num_neighbors=[16] * n_layers,
        batch_size=train_batch_size,
        input_nodes=train_idx.cpu(),
        shuffle=True,
        num_workers=4,
    )

    eval_loader = NeighborLoader(
        data,
        num_neighbors=[32] * n_layers,
        batch_size=32768,
        input_nodes=torch.cat([train_idx.cpu(), val_idx.cpu(), test_idx.cpu()]),
        shuffle=False,
        num_workers=4,
    )

    criterion    = nn.BCEWithLogitsLoss()
    model        = gen_model_fn().to(device)
    optimizer    = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.75, patience=50, verbose=True
    )

    total_time = 0
    best_val_score, final_test_score = 0, 0
    val_score  = 0
    final_pred = None

    for epoch in range(1, n_epochs + 1):
        tic  = time.time()
        loss = train_epoch(model, train_loader, criterion, optimizer, device, use_labels, n_classes)
        toc  = time.time()
        total_time += toc - tic

        if epoch == n_epochs or epoch % eval_every == 0 or epoch % log_every == 0:
            train_score, val_score, test_score, train_loss, val_loss, test_loss, pred = evaluate(
                model, eval_loader, labels, train_idx, val_idx, test_idx,
                criterion, evaluator_wrapper, device, use_labels, n_classes
            )

            if val_score > best_val_score:
                best_val_score   = val_score
                final_test_score = test_score
                final_pred       = pred

            if epoch % log_every == 0:
                print(
                    f'Epoch: {epoch:04d} | '
                    f'Loss: {loss:.4f} | '
                    f'Train: {100 * train_score:.2f}% | '
                    f'Valid: {100 * val_score:.2f}% | '
                    f'Test: {100 * test_score:.2f}% | '
                    f'Best Valid: {100 * best_val_score:.2f}% | '
                    f'Best Test: {100 * final_test_score:.2f}%'
                )

        lr_scheduler.step(val_score)

    if save_pred and final_pred is not None:
        os.makedirs('./output', exist_ok=True)
        torch.save(torch.sigmoid(final_pred), f'./output/{n_running}.pt')

    return best_val_score, final_test_score
