import torch
import torch.nn.functional as F
from src.model import StockPredictor
from src.data import StockNetDataModule
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os
import shutil
import json
from datetime import datetime


class ClassificationStockPredictor(StockPredictor):
    """三分类股票预测模型 - 预测涨跌平"""
    
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
        
        # 将连续标签转换为分类标签（3天平均收益的阈值）
        class_targets = torch.zeros_like(targets_flat, dtype=torch.long)
        class_targets[targets_flat > 1.0] = 0  # 涨（3天平均涨幅>1%）
        class_targets[targets_flat < -1.0] = 1  # 跌（3天平均跌幅>1%）
        class_targets[(targets_flat >= -1.0) & (targets_flat <= 1.0)] = 2  # 平（-1%到1%之间）
        
        # 计算交叉熵损失
        loss = F.cross_entropy(predictions_flat, class_targets)
        
        # 计算准确率
        pred_classes = torch.argmax(predictions_flat, dim=1)
        accuracy = (pred_classes == class_targets).float().mean()
        
        # 计算每个类别的准确率
        class_accuracies = []
        for i in range(3):
            mask = (class_targets == i)
            if mask.sum() > 0:
                class_acc = (pred_classes[mask] == class_targets[mask]).float().mean()
                class_accuracies.append(class_acc.item())
            else:
                class_accuracies.append(0.0)
        
        # 记录指标
        self.log(f"{mode}_loss", loss.item())
        self.log(f"{mode}_accuracy", accuracy.item())
        self.log(f"{mode}_class_0_acc", class_accuracies[0])  # 涨的准确率
        self.log(f"{mode}_class_1_acc", class_accuracies[1])  # 跌的准确率
        self.log(f"{mode}_class_2_acc", class_accuracies[2])  # 平的准确率
        
        return loss


def save_model_for_inference(model, save_dir="saved_models_classification_3day"):
    """保存模型用于推理"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存模型权重
    model_path = os.path.join(save_dir, "classification_model_3day.pth")
    torch.save(model.state_dict(), model_path)
    print(f"💾 模型权重已保存到: {model_path}")
    
    # 保存模型信息
    model_info = {
        "model_type": "ClassificationStockPredictor",
        "description": "三分类股票预测模型（涨/跌/平）- 3天窗口",
        "input_features": 6,  # OHLCV + 技术指标
        "embedding_dim": 768,  # BERT嵌入维度
        "hidden_dim": 128,     # 隐藏层维度
        "num_classes": 3,      # 分类数量
        "context_window": 3,  # 时间窗口
        "label_type": "3day_avg",  # 标签类型
        "classification_thresholds": {
            "up": 1.0,      # 涨类阈值
            "down": -1.0,   # 跌类阈值
            "flat": [-1.0, 1.0]  # 平类阈值范围
        },
        "class_names": ["涨", "跌", "平"],
        "class_descriptions": [
            "3天平均收益 > 1%",
            "3天平均收益 < -1%", 
            "3天平均收益在 -1% 到 1% 之间"
        ],
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_parameters": model.count_parameters()
    }
    
    info_path = os.path.join(save_dir, "model_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    print(f"📄 模型信息已保存到: {info_path}")
    
    return model_path, info_path


def create_inference_script(save_dir="saved_models_classification_3day"):
    """创建推理脚本"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_path = os.path.join(save_dir, f"inference_classification_3day_{timestamp}.py")
    
    script_content = '''import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from src.model import StockPredictor


class ClassificationStockPredictor(StockPredictor):
    """三分类股票预测模型：涨/跌/平"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 修改输出层为3分类
        self.head = nn.Sequential(
            nn.Linear(self.d_hidden, self.d_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.d_hidden // 2, 3)  # 3分类：涨/跌/平
        )
    
    def predict(self, x_ts, x_emb, x_id):
        """预测函数"""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x_ts, x_emb, x_id)
            probabilities = F.softmax(predictions, dim=-1)
            predicted_classes = torch.argmax(predictions, dim=-1)
            return predicted_classes, probabilities


def load_model(model_path, model_info_path):
    """加载模型"""
    # 读取模型信息
    with open(model_info_path, 'r', encoding='utf-8') as f:
        model_info = json.load(f)
    
    # 创建模型
    model = ClassificationStockPredictor(
        n_features=model_info["input_features"],
        d_emb=model_info["embedding_dim"],
        d_hidden=model_info["hidden_dim"],
        max_ids=87,
        n_blocks=4,
        d_head=model_info["hidden_dim"] // 4,
        n_head=4,
        dropout=0.3,
        use_linear_att=True,
        rotary_emb_list=[2],
        ignore_list=None,
        mode=0,
        lr=5e-5,
        weight_decay=1e-4
    )
    
    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model, model_info


def predict_stock_movement(model, model_info, timeseries_data, embedding_data, stock_id):
    """预测股票走势"""
    # 数据预处理（这里需要根据实际数据格式调整）
    # timeseries_data: [context_window, n_features]
    # embedding_data: [context_window, embedding_dim]
    # stock_id: 股票ID
    
    # 转换为tensor
    x_ts = torch.tensor(timeseries_data, dtype=torch.float32).unsqueeze(0)  # [1, context_window, n_features]
    x_emb = torch.tensor(embedding_data, dtype=torch.float32).unsqueeze(0)  # [1, context_window, embedding_dim]
    x_id = torch.tensor([stock_id], dtype=torch.long)  # [1]
    
    # 预测
    predicted_classes, probabilities = model.predict(x_ts, x_emb, x_id)
    
    # 获取结果
    class_idx = predicted_classes.item()
    class_name = model_info["class_names"][class_idx]
    class_description = model_info["class_descriptions"][class_idx]
    confidence = probabilities[0][class_idx].item()
    
    return {
        "prediction": class_name,
        "class_index": class_idx,
        "description": class_description,
        "confidence": confidence,
        "probabilities": {
            "涨": probabilities[0][0].item(),
            "跌": probabilities[0][1].item(),
            "平": probabilities[0][2].item()
        }
    }


def main():
    """主函数 - 示例用法"""
    # 模型路径
    model_path = "classification_model_3day.pth"
    model_info_path = "model_info.json"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    # 加载模型
    print("🤖 加载三分类模型（3天窗口）...")
    model, model_info = load_model(model_path, model_info_path)
    print("✅ 模型加载成功！")
    
    # 打印模型信息
    print(f"📊 模型类型: {model_info['model_type']}")
    print(f"📊 模型描述: {model_info['description']}")
    print(f"📊 训练日期: {model_info['training_date']}")
    print(f"📊 模型参数: {model_info['model_parameters']:,}")
    
    # 示例预测（需要实际数据）
    print("\\n📈 示例预测:")
    print("注意：需要提供实际的timeseries_data、embedding_data和stock_id")
    print("调用 predict_stock_movement(model, model_info, timeseries_data, embedding_data, stock_id)")
    
    # 示例数据结构
    print("\\n📋 输入数据格式:")
    print("- timeseries_data: [3, 6] - 3天窗口，6个特征（OHLCV+技术指标）")
    print("- embedding_data: [3, 768] - 3天窗口，768维BERT嵌入")
    print("- stock_id: 整数 - 股票ID")


if __name__ == "__main__":
    main()
'''
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"📝 推理脚本已创建: {script_path}")
    return script_path


def train_model():
    """训练三分类模型"""
    print("开始训练三分类模型（3天窗口）...")
    
    # 配置
    d_hidden = 128
    n_head = 4
    n_blocks = 4
    dropout = 0.3
    learning_rate = 5e-5
    batch_size = 16
    context_window = 3  # 使用3天窗口
    label_type = '3day_avg'  # 使用3天平均收益
    
    data_path = 'D:/Multimodal-Stock-Prediction2'
    
    # 使用改进的嵌入文件
    improved_embeddings_path = os.path.join(data_path, 'stocknet-dataset', 'embeddings', 'google-bert-bert-base-uncased_improved.npy')
    if os.path.exists(improved_embeddings_path):
        print("📁 使用改进的嵌入文件...")
        original_embeddings_path = os.path.join(data_path, 'all_embeddings.npy')
        shutil.copy2(improved_embeddings_path, original_embeddings_path)
        print("✅ 改进的嵌入文件已复制到原始位置")
    else:
        print("⚠️  未找到改进的嵌入文件，使用原始嵌入")
    
    # 使用修改后的数据模块
    datamod = StockNetDataModule(
        data_path=data_path,
        context_window=context_window,
        batch_size=batch_size,
        min_active_stock=1,
        a_threshold=0.0002,
        b_threshold=0.0055,
        num_workers=2,
        label_type=label_type  # 使用3天平均收益
    )
    
    # 创建三分类模型
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
    
    print(f'📈 模型参数数量: {model.count_parameters():,}')
    
    # 检查点回调
    experiment_name = "classification_model_3day"
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"experiments/{experiment_name}/weights",
        filename="best-epoch={epoch:02d}-val_accuracy={val_accuracy:.4f}",
        save_top_k=3,
        monitor="val_accuracy",
        mode="max"
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_accuracy", 
        min_delta=0.001, 
        patience=20, 
        verbose=True, 
        mode="max"
    )
    
    trainer = L.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=0.5,
        enable_progress_bar=True,
        log_every_n_steps=10,
        accumulate_grad_batches=1,
        precision="16-mixed" if torch.cuda.is_available() else "32"
    )
    
    # 开始训练
    print("🚀 开始训练三分类模型（3天窗口）...")
    print(f"📊 标签类型: {label_type}")
    print(f"🎯 分类阈值: 涨(>1%), 跌(<-1%), 平(-1%~1%)")
    
    # 训练模型
    trainer.fit(model, datamodule=datamod)
    
    # 测试模型
    print("🧪 测试模型...")
    results = trainer.test(ckpt_path='best', datamodule=datamod)
    
    # 输出结果
    print("\n" + "="*60)
    print("🎉 三分类模型训练完成（3天窗口）！")
    print("="*60)
    
    test_result = results[0]
    print(f"📊 测试准确率: {test_result['test_accuracy']:.4f}")
    print(f"📊 测试损失: {test_result['test_loss']:.4f}")
    print(f"📊 涨类准确率: {test_result['test_class_0_acc']:.4f}")
    print(f"📊 跌类准确率: {test_result['test_class_1_acc']:.4f}")
    print(f"📊 平类准确率: {test_result['test_class_2_acc']:.4f}")
    
    best_model_path = checkpoint_callback.best_model_path
    print(f"💾 最佳模型保存路径: {best_model_path}")
    
    # 保存模型用于推理
    print("\n💾 保存模型用于推理...")
    # 从最佳检查点加载模型
    best_model = ClassificationStockPredictor.load_from_checkpoint(
        best_model_path,
        n_features=6,
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
    save_model_for_inference(best_model)
    create_inference_script()
    
    return results, best_model_path


def main():
    """主函数"""
    print("🎯 三分类股票预测模型训练（3天窗口）")
    print("="*50)
    print("📋 配置:")
    print("   - 标签: 3天平均收益")
    print("   - 分类: 涨/跌/平 (3类)")
    print("   - 窗口: 3天")
    print("   - 阈值: ±1%")
    print("="*50)
    
    try:
        result, model_path = train_model()
        print(f"\n✅ 模型训练完成！最佳模型保存在: {model_path}")
        print(f"📊 测试结果: {result}")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 