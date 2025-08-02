from datetime import datetime, timedelta
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from sklearn.preprocessing import StandardScaler
import os
import json

class ImprovedStockNetDataModule(L.LightningDataModule):
    """改进的数据模块 - 包含更好的数据预处理和标签生成"""
    
    def __init__(
        self, 
        data_path, 
        context_window=32,
        batch_size=16, 
        min_active_stock=10, 
        a_threshold=0.,
        b_threshold=0.,
        num_workers=2,
        label_method='original',  # 标签生成方法
        embedding_threshold=0.1   # 嵌入质量阈值
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.context_window = context_window
        self.num_workers = num_workers
        self.min_active_stock = min_active_stock
        self.a_threshold = a_threshold
        self.b_threshold = b_threshold
        self.label_method = label_method
        self.embedding_threshold = embedding_threshold
        self.datasets = {}

    def prepare_data_per_node(self):
        pass

    def prepare_data(self):
        pass

    def _log_hyperparams(self, trainer):
        pass

    def setup(self, stage=None):
        print("设置改进的数据模块...")
        
        # 加载数据
        embeddings = np.load(f'{self.data_path}/all_embeddings.npy')
        timeseries = np.load(f'{self.data_path}/all_timeseries.npy')
        
        print(f"修复嵌入数据质量...")
        embeddings = self._fix_embedding_quality(embeddings)
        
        print(f"生成改进标签 (方法: {self.label_method})...")
        labels = self._generate_improved_labels(timeseries)
        
        print(f"标准化时间序列数据...")
        timeseries = self._normalize_timeseries(timeseries)
        
        # 时间分割
        s1 = datetime.strptime('2014-01-01', "%Y-%m-%d")
        s2 = datetime.strptime('2015-08-01', "%Y-%m-%d")
        s3 = datetime.strptime('2015-10-01', "%Y-%m-%d")
        t = (s2 - s1).days
        v = (s3 - s2).days

        # 创建数据集
        self.datasets['train'] = ImprovedStockNetDataset(
            embeddings=embeddings[:, :t],
            timeseries=timeseries[:, :t],
            labels=labels[:, :t],
            context_window=self.context_window,
            start_date='2014-01-01',
            min_active_stock=self.min_active_stock,
            a_threshold=self.a_threshold,
            b_threshold=self.b_threshold,
            label_method=self.label_method
        )
        
        self.datasets['val'] = ImprovedStockNetDataset(
            embeddings=embeddings[:, t:t+v],
            timeseries=timeseries[:, t:t+v],
            labels=labels[:, t:t+v],
            context_window=self.context_window,
            start_date='2015-08-01',
            min_active_stock=self.min_active_stock,
            a_threshold=self.a_threshold,
            b_threshold=self.b_threshold,
            label_method=self.label_method
        )
        
        self.datasets['test'] = ImprovedStockNetDataset(
            embeddings=embeddings[:, t+v:],
            timeseries=timeseries[:, t+v:],
            labels=labels[:, t+v:],
            context_window=self.context_window,
            start_date='2015-10-01',
            min_active_stock=self.min_active_stock,
            a_threshold=self.a_threshold,
            b_threshold=self.b_threshold,
            label_method=self.label_method
        )

    def _fix_embedding_quality(self, embeddings):
        """修复嵌入数据质量"""
        print(f"  原始零嵌入比例: {(embeddings.sum(axis=2) == 0).mean() * 100:.2f}%")
        
        # 计算嵌入质量
        embedding_norms = np.linalg.norm(embeddings, axis=2)
        low_quality_mask = embedding_norms < self.embedding_threshold
        print(f"  低质量嵌入比例: {low_quality_mask.mean() * 100:.2f}%")
        
        # 使用公司平均嵌入填充零嵌入
        for stock_idx in range(embeddings.shape[0]):
            stock_embeddings = embeddings[stock_idx]
            zero_mask = (stock_embeddings.sum(axis=1) == 0)
            
            if zero_mask.any():
                # 计算该公司的平均嵌入
                non_zero_embeddings = stock_embeddings[~zero_mask]
                if len(non_zero_embeddings) > 0:
                    avg_embedding = non_zero_embeddings.mean(axis=0)
                    embeddings[stock_idx, zero_mask] = avg_embedding
                else:
                    # 如果没有非零嵌入，使用全局平均
                    global_avg = embeddings[embeddings.sum(axis=2) != 0].mean(axis=0)
                    embeddings[stock_idx, zero_mask] = global_avg
        
        print(f"  修复后零嵌入比例: {(embeddings.sum(axis=2) == 0).mean() * 100:.2f}%")
        return embeddings

    def _generate_improved_labels(self, timeseries):
        """生成改进的标签"""
        prices = timeseries[:, :, 0]  # 收盘价
        labels = np.zeros_like(prices)
        
        for stock_idx in range(prices.shape[0]):
            stock_prices = prices[stock_idx]
            
            if self.label_method == 'original':
                # 原始方法：1天收益率
                for i in range(len(stock_prices) - 1):
                    if stock_prices[i] > 0:
                        labels[stock_idx, i] = (stock_prices[i+1] - stock_prices[i]) / stock_prices[i] * 100
                    else:
                        labels[stock_idx, i] = 0.0
                        
            elif self.label_method == 'improved':
                # 改进方法：对数收益率
                for i in range(len(stock_prices) - 1):
                    if stock_prices[i] > 0 and stock_prices[i+1] > 0:
                        labels[stock_idx, i] = np.log(stock_prices[i+1] / stock_prices[i]) * 100
                    else:
                        labels[stock_idx, i] = 0.0
                        
            elif self.label_method == 'log_return':
                # 对数收益率方法
                for i in range(len(stock_prices) - 1):
                    if stock_prices[i] > 0 and stock_prices[i+1] > 0:
                        labels[stock_idx, i] = np.log(stock_prices[i+1] / stock_prices[i]) * 100
                    else:
                        labels[stock_idx, i] = 0.0
        
        # 统计标签分布
        valid_labels = labels[labels != 0]
        if len(valid_labels) > 0:
            print(f"  标签统计:")
            print(f"    均值: {valid_labels.mean():.4f}%")
            print(f"    标准差: {valid_labels.std():.4f}%")
            print(f"    最小值: {valid_labels.min():.4f}%")
            print(f"    最大值: {valid_labels.max():.4f}%")
            print(f"    中位数: {np.median(valid_labels):.4f}%")
            
            # 分类统计
            flat_labels = valid_labels.flatten()
            flat_count = len(flat_labels)
            up_count = (flat_labels > 0.5).sum()
            down_count = (flat_labels < -0.5).sum()
            flat_count = ((flat_labels >= -0.5) & (flat_labels <= 0.5)).sum()
            
            print(f"    平盘样本: {flat_count} ({flat_count/flat_count*100:.2f}%)")
            print(f"    上涨样本: {up_count} ({up_count/flat_count*100:.2f}%)")
            print(f"    下跌样本: {down_count} ({down_count/flat_count*100:.2f}%)")
            
            # 异常值统计
            outlier_count = ((flat_labels > 20) | (flat_labels < -20)).sum()
            print(f"    异常值数量: {outlier_count} ({outlier_count/flat_count*100:.2f}%)")
        
        return labels

    def _normalize_timeseries(self, timeseries):
        """标准化时间序列数据"""
        # 只对训练集进行标准化
        train_timeseries = timeseries[:, :int(timeseries.shape[1] * 0.7)]
        
        for stock_idx in range(timeseries.shape[0]):
            stock_data = timeseries[stock_idx]
            train_data = train_timeseries[stock_idx]
            
            # 找到非零数据
            non_zero_mask = np.any(train_data != 0, axis=1)
            
            if non_zero_mask.sum() > 0:
                # 只对非零数据进行标准化
                stock_data_2d = train_data[non_zero_mask].reshape(-1, train_data.shape[1])
                scaler = StandardScaler()
                normalized_data = scaler.fit_transform(stock_data_2d)
                
                # 将标准化后的数据赋值回原数组
                timeseries[stock_idx, non_zero_mask] = normalized_data
        
        return timeseries

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(self._val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self._test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    @property
    def _train_dataset(self):
        return self.datasets['train']

    @property
    def _val_dataset(self):
        return self.datasets['val']

    @property
    def _test_dataset(self):
        return self.datasets['test']


class ImprovedStockNetDataset(Dataset):
    """改进的数据集 - 包含更好的数据处理"""
    
    def __init__(
        self, embeddings, timeseries, labels, context_window, start_date, 
        min_active_stock=1, a_threshold=0., b_threshold=0., label_method='original'
    ):
        super().__init__()
        self.embeddings = torch.from_numpy(embeddings).float()
        self.timeseries = torch.from_numpy(timeseries).float()
        self.labels = torch.from_numpy(labels).float()
        self.context_window = context_window
        self.start_date = start_date
        self.min_active_stock = min_active_stock
        self.a_threshold = a_threshold
        self.b_threshold = b_threshold
        self.label_method = label_method
        
        # 创建股票ID
        self.ids = torch.arange(embeddings.shape[0])
        
        # 数据过滤
        self._filter_data()
        
        print(f"数据集加载完成: {self._get_stage_name()}")
        print(f"  样本数量: {len(self)}")
        print(f"  时间窗口: {self.context_window}")
        print(f"  标签方法: {self.label_method}")

    def _get_stage_name(self):
        """获取数据集阶段名称"""
        if '2014-01-01' in self.start_date:
            return 'train'
        elif '2015-08-01' in self.start_date:
            return 'val'
        else:
            return 'test'

    def _filter_data(self):
        """过滤数据"""
        # 过滤活跃股票
        stock_mask = (self.timeseries.sum(dim=(1, 2)) != 0)
        self.timeseries = self.timeseries[stock_mask]
        self.embeddings = self.embeddings[stock_mask]
        self.labels = self.labels[stock_mask]
        self.ids = self.ids[stock_mask]
        
        # 过滤时间窗口
        time_mask = (self.timeseries.sum(dim=(0, 2)) != 0)
        self.timeseries = self.timeseries[:, time_mask]
        self.embeddings = self.embeddings[:, time_mask]
        self.labels = self.labels[:, time_mask]

    def __getitem__(self, idx):
        ts = self.timeseries[:, idx:idx + self.context_window]
        emb = self.embeddings[:, idx:idx + self.context_window]
        label = self.labels[:, idx + self.context_window - 1]  # 使用下一个时间点的标签
        
        # 处理空数据
        if ts.shape[1] == 0 or emb.shape[1] == 0:
            ts = torch.zeros((self.timeseries.shape[0], self.context_window, self.timeseries.shape[2]))
            emb = torch.zeros((self.embeddings.shape[0], self.context_window, self.embeddings.shape[2]))
            label = torch.zeros(self.timeseries.shape[0], dtype=torch.float)
        
        # 处理异常值
        label = torch.where(
            torch.isnan(label) | torch.isinf(label),
            torch.zeros_like(label),
            label
        )
        
        # 根据标签方法进行不同的处理
        if self.label_method == 'original':
            label = torch.clamp(label, -20.0, 20.0)
        elif self.label_method == 'improved':
            label = torch.clamp(label, -15.0, 15.0)
        elif self.label_method == 'log_return':
            label = torch.clamp(label, -10.0, 10.0)
        else:
            label = torch.clamp(label, -20.0, 20.0)

        return ts.float(), emb.float(), self.ids.long(), label.float()

    def __len__(self):
        return self.timeseries.shape[1] - self.context_window 