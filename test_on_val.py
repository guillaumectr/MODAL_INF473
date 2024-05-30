import torch
import wandb
import hydra
from tqdm import tqdm
import os

@hydra.main(config_path="configs/train", config_name="config")
def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    checkpoint_path = "/users/eleves-b/2022/guillaume.coutiere/MODAL/MODAL_INF473/checkpoints/dinov2_all_epochs_batch_500_dataaug_lr1em4_bs128_20epochs.pt"
    checkpoint_path = "/users/eleves-b/2022/guillaume.coutiere/MODAL/MODAL_INF473/checkpoints/dinov2_multi_situation_randomized_20_epochs.pt"
    checkpoint = torch.load(checkpoint_path)
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model.dino.load_state_dict(checkpoint)
    
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    val_loaders = datamodule.val_dataloader()

    for val_set_name, val_loader in val_loaders.items():
        epoch_num_correct = 0
        num_samples = 0
        y_true = []
        y_pred = []
        for i, batch in enumerate(val_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds.argmax(1).detach().cpu().tolist())
            epoch_num_correct += (
                (preds.argmax(1) == labels).sum().detach().cpu().numpy()
            )
            num_samples += len(images)
        epoch_acc = epoch_num_correct / num_samples
        print(f"{val_set_name} : {epoch_acc}")

if __name__ == "__main__":
    train()