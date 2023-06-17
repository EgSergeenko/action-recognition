import torch
import wandb
from sklearn import metrics
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class MetricsLogger(object):
    metrics = {
        'Accuracy, {0}': metrics.accuracy_score,
        'Precision, {0}': metrics.precision_score,
        'Recall, {0}': metrics.recall_score,
        'F1-Score, {0}': metrics.f1_score,
    }

    def __init__(self) -> None:
        self.logs: dict[str, list] = {}

    def log_epoch(
        self, y_true: int, y_pred: int, tag: str,
    ) -> None:
        step_log = {}
        for metric_name, metric in self.metrics.items():
            if 'Accuracy' in metric_name:
                score = metric(y_true, y_pred)
            else:
                score = metric(y_true, y_pred, average='macro')
            step_log[metric_name.format(tag)] = score
        if tag not in self.logs:
            self.logs[tag] = []
        self.logs[tag].append(step_log)

    def log_wandb(self) -> None:
        wandb_log = {}
        for tag in self.logs:
            step_log = self.logs[tag][-1]
            wandb_log.update(step_log)
        wandb.log(wandb_log)


def train_step(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    criterion: torch.nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> tuple[list[int], list[int], float]:
    optimizer.zero_grad()

    x, labels = batch[0].to(device), batch[1].to(device)

    output = model(x)

    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

    predictions = output.softmax(axis=1).argmax(axis=1).detach().cpu().tolist()
    labels = labels.argmax(axis=1).detach().cpu().tolist()

    return labels, predictions, loss.item()


def train_epoch(
    dataloader: DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> tuple[list[int], list[int], float]:
    model.train()
    epoch_loss = 0
    y_true, y_pred = [], []
    for batch in dataloader:
        target, prediction, step_loss = train_step(
            model=model,
            batch=batch,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        y_true.extend(target)
        y_pred.extend(prediction)
        epoch_loss += step_loss
    return y_true, y_pred, epoch_loss


@torch.no_grad()
def val_step(
    model: torch.nn.Module,
    sample: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[int, int]:
    x, label = sample[0].to(device).squeeze(), sample[1].to(device)

    output = model(x)
    predictions = output.softmax(axis=1).argmax(axis=1)
    prediction = predictions.mode()[0].item()

    return label.argmax().item(), prediction


def val_epoch(
    dataloader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
):
    model.eval()
    y_true, y_pred = [], []
    for batch in dataloader:
        target, prediction = val_step(
            model=model,
            sample=batch,
            device=device,
        )
        y_true.append(target)
        y_pred.append(prediction)
    return y_true, y_pred
