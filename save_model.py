import torch
import os
import json
from datetime import datetime
from src.model import StockPredictor
from src.data import StockNetDataModule
import lightning as L

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

def load_best_model(model_path):
    """加载最佳模型"""
    print(f"正在加载模型: {model_path}")
    
    # 模型配置
    model_config = {
        'n_features': 6,
        'd_emb': 768,
        'd_hidden': 128,
        'max_ids': 87,
        'n_blocks': 4,
        'd_head': 32,
        'n_head': 4,
        'dropout': 0.3,
        'use_linear_att': True,
        'rotary_emb_list': [2],
        'ignore_list': None,
        'mode': 0,
        'lr': 5e-5,
        'weight_decay': 1e-4
    }
    
    # 创建模型
    model = ClassificationStockPredictor(**model_config)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    print("✅ 模型加载成功！")
    return model

def save_model_for_inference(model, save_dir="saved_models"):
    """保存模型用于推理"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存模型文件
    model_filename = f"stock_predictor_{timestamp}.pth"
    model_path = os.path.join(save_dir, model_filename)
    
    # 保存模型状态
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'n_features': 6,
            'd_emb': 768,
            'd_hidden': 128,
            'max_ids': 87,
            'n_blocks': 4,
            'd_head': 32,
            'n_head': 4,
            'dropout': 0.3,
            'use_linear_att': True,
            'rotary_emb_list': [2],
            'ignore_list': None,
            'mode': 0
        },
        'model_type': 'classification',
        'num_classes': 3,
        'class_names': ['涨', '跌', '平'],
        'accuracy': 0.6809,  # 测试准确率
        'save_time': timestamp
    }, model_path)
    
    print(f"✅ 模型已保存到: {model_path}")
    
    # 保存模型信息
    info_filename = f"model_info_{timestamp}.json"
    info_path = os.path.join(save_dir, info_filename)
    
    model_info = {
        'model_path': model_path,
        'model_type': 'classification',
        'num_classes': 3,
        'class_names': ['涨', '跌', '平'],
        'accuracy': 0.6809,
        'save_time': timestamp,
        'model_config': {
            'n_features': 6,
            'd_emb': 768,
            'd_hidden': 128,
            'max_ids': 87,
            'n_blocks': 4,
            'd_head': 32,
            'n_head': 4,
            'dropout': 0.3
        },
        'usage_instructions': {
            'input_format': '时间序列数据 (OHLCV) + BERT嵌入 + 股票ID',
            'output_format': '3分类概率 (涨/跌/平)',
            'prediction_threshold': '0.5% 变化阈值'
        }
    }
    
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 模型信息已保存到: {info_path}")
    
    return model_path, info_path

def create_inference_script(save_dir="saved_models"):
    """创建推理脚本"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_filename = f"inference_{timestamp}.py"
    script_path = os.path.join(save_dir, script_filename)
    
    script_content = '''import torch
import numpy as np
from src.model import StockPredictor
from src.data import StockNetDataModule

class ClassificationStockPredictor(StockPredictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.d_hidden, self.d_hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(self.d_hidden // 2, 3)
        )

def load_model(model_path):
    """加载保存的模型"""
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['model_config']
    
    model = ClassificationStockPredictor(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def predict_stock(model, x_ts, x_emb, x_id):
    """预测股票涨跌"""
    with torch.no_grad():
        predictions = model(x_ts, x_emb, x_id)
        probabilities = torch.softmax(predictions, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1)
        
    return {
        'probabilities': probabilities.numpy(),
        'predicted_class': predicted_class.numpy(),
        'class_names': ['涨', '跌', '平']
    }

# 使用示例
if __name__ == "__main__":
    # 加载模型
    model_path = "stock_predictor_20241220_123456.pth"  # 替换为实际路径
    model = load_model(model_path)
    
    # 准备输入数据（这里需要根据实际情况调整）
    # x_ts: 时间序列数据 [batch_size, num_stocks, seq_len, features]
    # x_emb: BERT嵌入 [batch_size, num_stocks, seq_len, 768]
    # x_id: 股票ID [batch_size, num_stocks]
    
    print("模型加载完成，可以开始预测！")
'''
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"✅ 推理脚本已创建: {script_path}")
    return script_path

def main():
    """主函数"""
    print("开始保存模型...")
    
    # 查找最佳模型路径
    model_paths = [
        "experiments/improved_data_direction/weights/best-epoch=01-val_accuracy=0.6809.ckpt",
        "experiments/improved_data_direction/weights/best-epoch=00-val_accuracy=0.6809.ckpt",
        "lightning_logs/version_*/checkpoints/*.ckpt"
    ]
    
    best_model_path = None
    for path in model_paths:
        if os.path.exists(path):
            best_model_path = path
            break
    
    if not best_model_path:
        print("❌ 未找到训练好的模型文件")
        print("请确保已经运行过 train_with_improved_data.py 并成功训练")
        return
    
    try:
        # 加载模型
        model = load_best_model(best_model_path)
        
        # 保存模型
        save_dir = "saved_models"
        model_path, info_path = save_model_for_inference(model, save_dir)
        
        # 创建推理脚本
        script_path = create_inference_script(save_dir)
        
        print("\n🎉 模型保存完成！")
        print(f"📁 保存目录: {save_dir}")
        print(f"📄 模型文件: {os.path.basename(model_path)}")
        print(f"📄 模型信息: {os.path.basename(info_path)}")
        print(f"📄 推理脚本: {os.path.basename(script_path)}")
        
        print("\n📋 使用说明:")
        print("1. 模型文件 (.pth) 包含了训练好的权重")
        print("2. 模型信息 (.json) 包含了模型配置和使用说明")
        print("3. 推理脚本 (.py) 提供了加载和预测的示例代码")
        print("4. 准确率: 68.09% (三分分类)")
        
    except Exception as e:
        print(f"❌ 保存模型失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 