import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.functional import mean_squared_error, mean_absolute_error, r2_score
import lightning as L

from .modules import RMSNorm
from .transformer import HighOrderTransformer


class StockPredictor(L.LightningModule):
    def __init__(
        self, 
        n_features,
        d_emb,
        d_hidden,
        max_ids = 2048,
        n_blocks = 2, 
        d_head = 16, 
        n_head = 4, 
        dropout=0., 
        use_linear_att=True,
        feature_map='SMReg',
        rotary_emb_list=None,
        ignore_list=None,
        mode=0,
        lr=1e-4,
        weight_decay=0.
    ):
        super().__init__()
        self.save_hyperparameters()
        self.d_hidden = d_hidden
        self.lr = lr
        self.weight_decay = weight_decay
        self.mode = mode # 1=ts-only, 2=emb-only , 3=multimodal

        self.ts_proj = nn.Linear(n_features, d_hidden)
        self.emb_proj = nn.Linear(d_emb, d_hidden)
        self.id_embedding = nn.Embedding(num_embeddings=max_ids, embedding_dim=d_hidden)
        self.mask_token = nn.Parameter(torch.zeros(d_hidden))

        self.encoder = HighOrderTransformer(
            d_hidden, 
            n_blocks, 
            d_head, 
            n_head, 
            dropout, 
            use_linear_att, 
            feature_map,
            rotary_emb_list, 
            ignore_list
        )
        self.decoder = HighOrderTransformer(
            d_hidden, 
            n_blocks, 
            d_head, 
            n_head, 
            dropout, 
            use_linear_att, 
            feature_map,
            rotary_emb_list, 
            ignore_list
        )        
        # 修改输出层为回归：预测涨跌百分比
        self.head = nn.Sequential(RMSNorm(d_hidden), nn.Linear(d_hidden, 1))
        self.dropout = nn.Dropout(p=dropout)
        
        
    def forward_encoder(self, x_emb, x_id):
        id_emb = self.id_embedding(x_id).unsqueeze(2)  # (bs, n, 1, d)
        h_emb = self.dropout(self.emb_proj(x_emb))    # (bs, n, t, d)
        # 修复数据类型不匹配问题
        mask_condition = h_emb.sum(3) == 0.
        h_emb[mask_condition] = self.mask_token.to(h_emb.dtype)   # replacing zero values with trainable mask token
        h_emb = torch.cat([id_emb, h_emb], dim=2)     # (bs, n, 1 + t, d)     
        return self.encoder(h_emb)

    def forward_decoder(self, x_ts, x_id, emb_hiddens=None):
        id_emb = self.id_embedding(x_id).unsqueeze(2)
        h_ts = self.dropout(self.ts_proj(x_ts))
        # 修复数据类型不匹配问题
        mask_condition = h_ts.sum(3) == 0.
        h_ts[mask_condition] = self.mask_token.to(h_ts.dtype)
        h_ts = torch.cat([id_emb, h_ts], dim=2)
        h_ts, _ = self.decoder(h_ts, emb_hiddens)
        return h_ts

    def forward(self, x_ts=None, x_emb=None, x_id=None):
        assert (x_id is not None) and (x_ts is not None or x_emb is not None), "Invalid inputs"
        emb_hiddens = None
        if self.mode > 0:
            h_emb, emb_hiddens = self.forward_encoder(x_emb, x_id)
            if self.mode == 1:
                return self.head(h_emb[:, :, 0, :])
        
        h_ts = self.forward_decoder(x_ts, x_id, emb_hiddens)
        return self.head(h_ts[:, :, 0, :])
    

    def calc_metrics(self, predictions, targets):
        """计算回归指标"""
        # 展平数据
        preds = predictions.flatten()
        targets = targets.flatten()
        
        # 过滤掉无效数据
        valid_mask = ~torch.isnan(targets) & ~torch.isinf(targets)
        if valid_mask.sum() == 0:
            # 如果没有有效数据，返回默认值
            device = predictions.device
            mse = torch.tensor(0.0, device=device)
            rmse = torch.tensor(0.0, device=device)
            mae = torch.tensor(0.0, device=device)
            r2 = torch.tensor(0.0, device=device)
            return mse, rmse, mae, r2
        
        preds = preds[valid_mask]
        targets = targets[valid_mask]
        
        # 计算回归指标
        mse = F.mse_loss(preds, targets)
        rmse = torch.sqrt(mse)
        mae = F.l1_loss(preds, targets)
        
        # 计算R²分数
        ss_res = torch.sum((targets - preds) ** 2)
        ss_tot = torch.sum((targets - targets.mean()) ** 2)
        if ss_tot == 0:
            r2 = torch.tensor(0.0, device=predictions.device)
        else:
            r2 = 1 - (ss_res / ss_tot)
        
        return mse, rmse, mae, r2
    

    def step(self, batch, mode='train'):
        x_ts, x_emb, x_id, targets = batch
        predictions = self.forward(x_ts, x_emb, x_id)
        mse, rmse, mae, r2 = self.calc_metrics(predictions, targets)

        # 记录指标
        self.log(f"{mode}_mse", mse.item())
        self.log(f"{mode}_rmse", rmse.item())
        self.log(f"{mode}_mae", mae.item())
        self.log(f"{mode}_r2", r2.item())
        
        # 使用MSE作为主要损失函数
        loss = mse
        self.log(f"{mode}_loss", loss.item())
        
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, mode='train')
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, mode='val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, mode='test')
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def configure_optimizers(self):
        # 回归任务使用更小的学习率
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr * 0.1, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
        return [optimizer]#, [lr_scheduler]

    