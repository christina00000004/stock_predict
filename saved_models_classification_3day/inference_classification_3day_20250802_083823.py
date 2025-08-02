import torch
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
    print("\n📈 示例预测:")
    print("注意：需要提供实际的timeseries_data、embedding_data和stock_id")
    print("调用 predict_stock_movement(model, model_info, timeseries_data, embedding_data, stock_id)")
    
    # 示例数据结构
    print("\n📋 输入数据格式:")
    print("- timeseries_data: [3, 6] - 3天窗口，6个特征（OHLCV+技术指标）")
    print("- embedding_data: [3, 768] - 3天窗口，768维BERT嵌入")
    print("- stock_id: 整数 - 股票ID")


if __name__ == "__main__":
    main()
