多模态股票预测模型
==================

项目简介
--------
这是一个基于深度学习的多模态股票预测模型，结合时间序列数据（OHLCV）和文本数据（Twitter推文）来预测股票涨跌方向。模型使用Transformer架构，支持多种时间窗口和标签类型。

核心文件
--------
- train_classification_model_3day.py - 三分类训练脚本（3天窗口）
- train_with_improved_data.py - 主训练脚本（三分类：涨/跌/平）
- save_model.py - 模型保存脚本
- src/ - 核心模型和数据模块
  - src/model.py - 模型定义
  - src/data.py - 数据处理模块
  - src/transformer.py - Transformer架构
  - src/modules.py - 基础模块

数据文件
--------
- all_embeddings.npy - BERT文本嵌入数据
- all_timeseries.npy - 时间序列数据
- stocknet-dataset/ - 原始数据集
  - stocknet-dataset/embeddings/google-bert-bert-base-uncased_improved.npy - 改进的嵌入文件

模型配置
--------
当前提供多种训练配置：

1. 3天窗口模型 (train_classification_model_3day.py)
   - 时间窗口: 3天
   - 标签类型: 3天平均收益
   - 分类阈值: ±1%
   - 保存目录: saved_models_classification_3day/

2. 5天窗口模型 (train_classification_model.py)
   - 时间窗口: 5天
   - 标签类型: 5天平均收益
   - 分类阈值: ±1%
   - 保存目录: saved_models_classification/

3. 10天窗口模型 (train_with_improved_data.py)
   - 时间窗口: 10天
   - 标签类型: 10天平均收益
   - 分类阈值: ±1%
   - 保存目录: experiments/improved_data_10day_avg/

模型性能
--------
- 准确率: 68.09% (三分类，基于10天平均收益)
- 预测类型: 涨/跌/平
- 数据时间: 2014-2016年
- 模型参数: 约400万参数
- 输入特征: 6维（OHLCV + 技术指标）
- 嵌入维度: 768维（BERT）

使用方法
--------

1. 训练3天窗口模型
   python train_classification_model_3day.py

2. 训练5天窗口模型
   python train_classification_model.py

3. 训练10天窗口模型
   python train_with_improved_data.py

4. 保存模型
   python save_model.py

模型输出
--------
- 0类：上涨（平均涨幅 > 1%）
- 1类：下跌（平均跌幅 < -1%）
- 2类：平盘（-1% ≤ 平均变化 ≤ 1%）

环境要求
--------
- Python 3.8+
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- Transformers (BERT)
- NumPy
- CUDA (推荐用于GPU加速)

数据预处理
--------
- 时间序列数据已标准化
- BERT嵌入已优化（处理零嵌入问题）
- 支持多种标签类型：1天收益、3天平均、5天平均、10天平均

注意事项
--------
- 模型基于历史数据训练，不保证未来预测准确性
- 需要GPU加速训练（推荐RTX 3070或更高）
- 数据预处理已完成，可直接运行训练脚本
- 改进的嵌入文件会自动复制到训练位置
- 训练完成后会自动保存模型和推理脚本

项目结构
--------
```
Multimodal-Stock-Prediction2/
├── train_classification_model_3day.py    # 3天窗口训练
├── train_with_improved_data.py           # 主训练脚本
├── save_model.py                         # 模型保存
├── src/                                  # 核心模块
├── experiments/                          # 实验记录
├── saved_models_classification_3day/     # 3天模型保存
├── saved_models_classification/          # 5天模型保存
└── stocknet-dataset/                     # 原始数据
``` 