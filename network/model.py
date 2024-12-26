import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


class MyModel(L.LightningModule):
    def __init__(self):
        # Dummy network
        self.network = nn.Identity()
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        predicted_y = self.network(x)
        loss = F.mse_loss(x. y)
        
        # Logging
        self.log("MSE Loss", loss, prog_bar=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        pass
        

    def test_step(self, batch, batch_idx):
        pass
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-8,
        )
        
        return [optimizer], []

        