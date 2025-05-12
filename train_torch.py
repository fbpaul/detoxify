import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.optim import Adam
from src.utils import get_model_and_tokenizer
import src.data_loaders as module_data

class ToxicClassifier(torch.nn.Module):
    """Toxic comment classification for the Jigsaw challenges."""

    def __init__(self, config):
        super().__init__()
        self.num_classes = config["arch"]["args"]["num_classes"]
        self.model_args = config["arch"]["args"]
        self.model, self.tokenizer = get_model_and_tokenizer(**self.model_args)
        self.loss_weight = config.get("loss_weight", None)
        self.num_main_classes = config.get("num_main_classes", self.num_classes)
        self.bias_loss = bool(self.num_main_classes < self.num_classes)

    def forward(self, x):
        inputs = self.tokenizer(x, return_tensors="pt", truncation=True, padding=True).to(self.model.device)
        outputs = self.model(**inputs)[0]
        return outputs

    def binary_cross_entropy(self, input, meta):
        if "weight" in meta:
            target = meta["target"].to(input.device).reshape(input.shape)
            weight = meta["weight"].to(input.device).reshape(input.shape)
            return F.binary_cross_entropy_with_logits(input, target, weight=weight)
        elif "multi_target" in meta:
            target = meta["multi_target"].to(input.device)
            loss_fn = F.binary_cross_entropy_with_logits
            mask = target != -1
            loss = loss_fn(input, target.float(), reduction="none")
            if "class_weights" in meta:
                weights = meta["class_weights"][0].to(input.device)
            elif "weights1" in meta:
                weights = meta["weights1"].to(input.device)
            else:
                weights = torch.tensor(1 / self.num_main_classes).to(input.device)
                loss = loss[:, : self.num_main_classes]
                mask = mask[:, : self.num_main_classes]
            weighted_loss = loss * weights
            nz = torch.sum(mask, 0) != 0
            masked_tensor = weighted_loss * mask
            masked_loss = torch.sum(masked_tensor[:, nz], 0) / torch.sum(mask[:, nz], 0)
            loss = torch.sum(masked_loss)
            return loss
        else:
            target = meta["target"].to(input.device)
            return F.binary_cross_entropy_with_logits(input, target.float())

    def binary_accuracy(self, output, meta):
        if "multi_target" in meta:
            target = meta["multi_target"].to(output.device)
        else:
            target = meta["target"].to(output.device)
        with torch.no_grad():
            mask = target != -1
            pred = torch.sigmoid(output[mask]) >= 0.5
            correct = torch.sum(pred.to(output[mask].device) == target[mask])
            if torch.sum(mask).item() != 0:
                correct = correct.item() / torch.sum(mask).item()
            else:
                correct = 0
        return torch.tensor(correct)

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        x, meta = batch
        optimizer.zero_grad()
        output = model(x)
        loss = model.binary_cross_entropy(output, meta)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for batch in val_loader:
            x, meta = batch
            output = model(x)
            loss = model.binary_cross_entropy(output, meta)
            acc = model.binary_accuracy(output, meta)
            total_loss += loss
            total_acc += acc
    return total_loss / len(val_loader), total_acc / len(val_loader)

def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cuda:1",
        type=str,
        help="comma-separated indices of GPUs to enable (default: None)",
    )
    parser.add_argument(
        "--num_workers",
        default=10,
        type=int,
        help="number of workers used in the data loader (default: 10)",
    )
    parser.add_argument("-e", "--n_epochs", default=100, type=int, help="if given, override the num")
    args = parser.parse_args()
    config = json.load(open(args.config))

    # 数据加载
    def get_instance(module, name, config, *args, **kwargs):
        return getattr(module, config[name]["type"])(*args, **config[name]["args"], **kwargs)

    dataset = get_instance(module_data, "dataset", config)
    val_dataset = get_instance(module_data, "dataset", config, train=False)

    data_loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    valid_data_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=args.num_workers,
        shuffle=False,
    )

    # 模型
    model = ToxicClassifier(config).to(args.device)
    optimizer = Adam(model.parameters(), **config["optimizer"]["args"])

    # 训练
    for epoch in range(args.n_epochs):
        train_loss = train(model, data_loader, optimizer, args.device)
        val_loss, val_acc = validate(model, valid_data_loader, args.device)
        print(f"Epoch {epoch + 1}/{args.n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

if __name__ == "__main__":
    main()