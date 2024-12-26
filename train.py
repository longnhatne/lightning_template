import lightning as L
from torch.utils.data import DataLoader, random_split
from data.data_zip import ImageFolderDataset, FrameVideoFolderDataset
from networks.generator import Generator
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
import os
from lightning.pytorch.callbacks import TQDMProgressBar

batch_size = 3
accumulation_step = 20

root_dir = "experiments" # log folder
resume_ckpt = None # checkpoint path, 

def main():
    L.seed_everything(666)

    dataset = None
    train_dataset, val_dataset  = random_split(dataset, [n-200, 200])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    model = None
    
    # Trainer
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2, # Save top k checkpoints
        monitor="global_step",
        mode="max",
    )
    trainer = L.Trainer(default_root_dir=root_dir,
                        callbacks=[checkpoint_callback, lr_monitor, TQDMProgressBar(refresh_rate=50)],
                        strategy='ddp_find_unused_parameters_true',
                        val_check_interval=0.25,
                        log_every_n_steps=50,
                        max_steps=10000000,
                        num_nodes=1)

    # Train
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=resume_ckpt)
    

if __name__ == "__main__":
    main()