import torch
import torch.nn.functional as F
from src.model import StockPredictor
from src.data import StockNetDataModule
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os

class ClassificationStockPredictor(StockPredictor):
    """分类股票预测模型 - 预测涨跌方向"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 修改输出层为3分类（涨、跌、平）
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.d_hidden, self.d_hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(self.d_hidden // 2, 3)  # 3分类
        )
    
    def step(self, batch, mode='train'):
        x_ts, x_emb, x_id, targets = batch
        predictions = self.forward(x_ts, x_emb, x_id)
        
        # 处理标签维度：targets是[batch_size, num_stocks]，需要展平
        batch_size, num_stocks = targets.shape
        targets_flat = targets.flatten()  # [batch_size * num_stocks]
        predictions_flat = predictions.view(-1, predictions.size(-1))  # [batch_size * num_stocks, num_classes]
        
        # 将连续标签转换为分类标签（10天平均收益的阈值调整）
        class_targets = torch.zeros_like(targets_flat, dtype=torch.long)
        class_targets[targets_flat > 1.0] = 0  # 涨（10天平均涨幅>1%）
        class_targets[targets_flat < -1.0] = 1  # 跌（10天平均跌幅>1%）
        class_targets[(targets_flat >= -1.0) & (targets_flat <= 1.0)] = 2  # 平（-1%到1%之间）
        
        # 计算交叉熵损失
        loss = F.cross_entropy(predictions_flat, class_targets)
        
        # 计算准确率
        pred_classes = torch.argmax(predictions_flat, dim=1)
        accuracy = (pred_classes == class_targets).float().mean()
        
        # 记录指标
        self.log(f"{mode}_loss", loss.item())
        self.log(f"{mode}_accuracy", accuracy.item())
        
        return loss

def train_model(label_type='direction'):
    """训练模型"""
    print(f"开始训练模型，标签类型: {label_type}")
    
    # 配置
    d_hidden = 128
    n_head = 4
    n_blocks = 4
    dropout = 0.3
    learning_rate = 5e-5
    batch_size = 16
    context_window = 10
    
    data_path = 'D:/Multimodal-Stock-Prediction2'
    
    # 使用改进的嵌入文件
    improved_embeddings_path = os.path.join(data_path, 'stocknet-dataset', 'embeddings', 'google-bert-bert-base-uncased_improved.npy')
    if os.path.exists(improved_embeddings_path):
        print("使用改进的嵌入文件...")
        original_embeddings_path = os.path.join(data_path, 'all_embeddings.npy')
        import shutil
        shutil.copy2(improved_embeddings_path, original_embeddings_path)
        print("改进的嵌入文件已复制到原始位置")
    
    # 使用修改后的数据模块
    datamod = StockNetDataModule(
        data_path=data_path,
        context_window=context_window,
        batch_size=batch_size,
        min_active_stock=1,
        a_threshold=0.0002,
        b_threshold=0.0055,
        num_workers=2,
        label_type=label_type  # 使用新的标签类型参数
    )
    
    # 根据标签类型选择模型
    if label_type == 'direction':
        model = ClassificationStockPredictor(
            n_features=6,  # 原始特征数量
            d_emb=768,
            d_hidden=d_hidden,
            max_ids=87,
            n_blocks=n_blocks,
            d_head=d_hidden // n_head,
            n_head=n_head,
            dropout=dropout,
            use_linear_att=True,
            rotary_emb_list=[2],
            ignore_list=None,
            mode=0,
            lr=learning_rate,
            weight_decay=1e-4
        )
        monitor_metric = "val_accuracy"
        mode = "max"
    else:
        # 使用原始的回归模型
        model = StockPredictor(
            n_features=6,  # 原始特征数量
            d_emb=768,
            d_hidden=d_hidden,
            max_ids=87,
            n_blocks=n_blocks,
            d_head=d_hidden // n_head,
            n_head=n_head,
            dropout=dropout,
            use_linear_att=True,
            rotary_emb_list=[2],
            ignore_list=None,
            mode=0,
            lr=learning_rate,
            weight_decay=1e-4
        )
        monitor_metric = "val_r2"
        mode = "max"
    
    print(f'模型参数数量: {model.count_parameters()}')
    
    # 检查点回调
    experiment_name = f"improved_data_{label_type}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"experiments/{experiment_name}/weights",
        filename=f"best-{{epoch:02d}}-{{{monitor_metric}:.4f}}",
        save_top_k=3,
        monitor=monitor_metric,
        mode=mode
    )
    
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric, 
        min_delta=0.001, 
        patience=20, 
        verbose=True, 
        mode=mode
    )
    
    trainer = L.Trainer(
        max_epochs=100,
        accelerator="gpu",
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=0.5,
        enable_progress_bar=True,
        log_every_n_steps=10,
        accumulate_grad_batches=1
    )
    
    # 训练模型
    trainer.fit(model, datamodule=datamod)
    
    # 测试模型
    results = trainer.test(ckpt_path='best', datamodule=datamod)
    
    return results, checkpoint_callback.best_model_path

def main():
    """主函数"""
    print("开始训练模型...")
    
    # 可选的标签类型
    label_types = ['direction', '5day_avg', '10day_avg', 'volatility', '1day_return']
    
    print("可选的标签类型:")
    print("1. direction - 方向分类（涨跌平）")
    print("2. 5day_avg - 5天平均收益")
    print("3. 10day_avg - 10天平均收益")
    print("4. volatility - 波动率")
    print("5. 1day_return - 1天收益率（原始）")
    
    # 使用10天平均收益（更稳定的标签）
    label_type = '10day_avg'
    
    try:
        result, model_path = train_model(label_type)
        print(f"✅ 模型训练完成！最佳模型保存在: {model_path}")
        print(f"测试结果: {result}")
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 