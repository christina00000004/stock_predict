from datetime import datetime, timedelta
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from sklearn.preprocessing import StandardScaler

#StockNetDataModule 类
#用于构建训练/验证/测试数据：
#构造函数 __init__
#data_path: .npy 数据所在路径
#context_window: 每个输入序列的时间窗口（比如用过去 32 天预测未来）
#min_active_stock: 每天至少有这么多只股票是"有效"的
#a_threshold / b_threshold: 标签分类阈值（上涨、下跌、中性）


class StockNetDataModule(L.LightningDataModule):
    def __init__(
        self, 
        data_path, 
        context_window=32,
        batch_size=16, 
        min_active_stock=10, 
        a_threshold=0.,
        b_threshold=0.,
        num_workers=2,
        label_type='1day_return'  # 新增：标签类型参数
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.context_window = context_window
        self.num_workers = num_workers
        self.min_active_stock = min_active_stock
        self.a_threshold = a_threshold
        self.b_threshold = b_threshold
        self.label_type = label_type  # 新增：标签类型
        self.datasets = {}

    def setup(self, stage=None):
        embeddings = torch.from_numpy(np.load(f'{self.data_path}/all_embeddings.npy'))
        timeseries = torch.from_numpy(np.load(f'{self.data_path}/all_timeseries.npy'))
        
        s1 = datetime.strptime('2014-01-01', "%Y-%m-%d")
        s2 = datetime.strptime('2015-08-01', "%Y-%m-%d")
        s3 = datetime.strptime('2015-10-01', "%Y-%m-%d")
        t = (s2 - s1).days
        v = (s3 - s2).days

        mu = timeseries[:, :t].mean(dim=(0,1))
        std = timeseries[:, :t].std(dim=(0,1))
        timeseries = (timeseries - mu) / std
        
        self.datasets['train'] = StockNetDataset(
            embeddings=embeddings[:, :t],
            timeseries=timeseries[:, :t],
            context_window=self.context_window,
            start_date='2014-01-01',
            min_active_stock=self.min_active_stock,
            a_threshold=self.a_threshold,
            b_threshold=self.b_threshold,
            label_type=self.label_type  # 新增：传递标签类型
        )
        self.datasets['val'] = StockNetDataset(
            embeddings=embeddings[:, t:t+v],
            timeseries=timeseries[:, t:t+v],
            context_window=self.context_window,
            start_date='2015-08-01',
            min_active_stock=self.min_active_stock,
            a_threshold=self.a_threshold,
            b_threshold=self.b_threshold,
            label_type=self.label_type  # 新增：传递标签类型
        )
        self.datasets['test'] = StockNetDataset(
            embeddings=embeddings[:, t+v:],
            timeseries=timeseries[:, t+v:],
            context_window=self.context_window,
            start_date='2015-10-01',
            min_active_stock=self.min_active_stock,
            a_threshold=self.a_threshold,
            b_threshold=self.b_threshold,
            label_type=self.label_type  # 新增：传递标签类型
        )
        

    def train_dataloader(self):
        return DataLoader(
            self.datasets['train'], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(self.datasets['val'], batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.datasets['test'], batch_size=self.batch_size, num_workers=self.num_workers)




class StockNetDataset(Dataset):
    def __init__(
        self, embeddings, timeseries, context_window, start_date, min_active_stock=1, a_threshold=0., b_threshold=0., label_type='1day_return'
    ):
        super().__init__()
        self.context_window = context_window
        self.a_threshold = a_threshold
        self.b_threshold = b_threshold
        self.label_type = label_type  # 新增：标签类型

        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        all_dates = []
        for i in range(timeseries.shape[1]):
            date = start_date + timedelta(days=i)
            all_dates += [torch.tensor([date.year, date.month, date.day])]
        all_dates = torch.stack(all_dates).unsqueeze(0).repeat(timeseries.shape[0], 1, 1)
        self.timeseries = torch.cat([timeseries, all_dates], dim=2)
        self.ids = torch.arange(self.timeseries.shape[0])
        
        mask = (embeddings.sum(2) != 0) & (timeseries.sum(2) != 0)
        time_mask = (mask.sum(0) > min_active_stock)
        stock_mask = (mask.sum(1) > context_window)
        self.timeseries = self.timeseries[stock_mask, :][:, time_mask]
        self.embeddings = embeddings[stock_mask, :][:, time_mask]
        self.ids = self.ids[stock_mask]

    def _calculate_label(self, idx):
        """根据标签类型计算标签"""
        current_price = self.timeseries[:, idx + self.context_window - 1, 0]  # 当前价格
        
        if self.label_type == 'direction':
            # 方向分类：1天后的涨跌方向
            if idx + self.context_window < self.timeseries.shape[1]:
                future_price = self.timeseries[:, idx + self.context_window, 0]
                return (future_price - current_price) / current_price * 100
            else:
                return torch.zeros_like(current_price)
                
        elif self.label_type == '5day_avg':
            # 5天平均收益
            if idx + self.context_window + 4 < self.timeseries.shape[1]:
                future_prices = self.timeseries[:, idx + self.context_window:idx + self.context_window + 5, 0]
                avg_future_price = future_prices.mean(dim=1)
                return (avg_future_price - current_price) / current_price * 100
            else:
                return torch.zeros_like(current_price)
                
        elif self.label_type == '10day_avg':
            # 10天平均收益
            if idx + self.context_window + 9 < self.timeseries.shape[1]:
                future_prices = self.timeseries[:, idx + self.context_window:idx + self.context_window + 10, 0]
                avg_future_price = future_prices.mean(dim=1)
                return (avg_future_price - current_price) / current_price * 100
            else:
                return torch.zeros_like(current_price)
                
        elif self.label_type == 'volatility':
            # 波动率（未来5天的标准差）
            if idx + self.context_window + 4 < self.timeseries.shape[1]:
                future_prices = self.timeseries[:, idx + self.context_window:idx + self.context_window + 5, 0]
                returns = (future_prices[:, 1:] - future_prices[:, :-1]) / future_prices[:, :-1] * 100
                volatility = torch.std(returns, dim=1)
                return volatility
            else:
                return torch.zeros_like(current_price)
        
        else:
            # 默认：1天收益率
            if idx + self.context_window < self.timeseries.shape[1]:
                future_price = self.timeseries[:, idx + self.context_window, 0]
                return (future_price - current_price) / current_price * 100
            else:
                return torch.zeros_like(current_price)


    def __getitem__(self, idx):
        ts = self.timeseries[:, idx:idx + self.context_window]
        emb = self.embeddings[:, idx:idx + self.context_window]
        
        # 检查数据是否为空
        if ts.shape[1] == 0 or emb.shape[1] == 0:
            # 如果数据为空，返回一个默认的样本
            ts = torch.zeros((self.timeseries.shape[0], self.context_window, self.timeseries.shape[2]))
            emb = torch.zeros((self.embeddings.shape[0], self.context_window, self.embeddings.shape[2]))
            # 回归标签：默认0%变化
            label = torch.zeros(self.timeseries.shape[0], dtype=torch.float)
            return ts.float(), emb.float(), self.ids.long(), label.float()
        
        # 使用新的标签计算方法
        label = self._calculate_label(idx)
        
        # 处理异常值
        label = torch.where(
            torch.isnan(label) | torch.isinf(label),
            torch.zeros_like(label),
            label
        )
        
        # 根据标签类型进行不同的处理
        if self.label_type == 'direction':
            # 方向分类：限制在[-1, 1]范围内
            label = torch.clamp(label, -1.0, 1.0)
        elif self.label_type in ['5day_avg', '10day_avg']:
            # 平均收益：限制在[-20, 20]范围内
            label = torch.clamp(label, -20.0, 20.0)
        elif self.label_type == 'volatility':
            # 波动率：限制在[0, 50]范围内
            label = torch.clamp(label, 0.0, 50.0)
        else:
            # 默认：限制在[-20, 20]范围内
            label = torch.clamp(label, -20.0, 20.0)

        return ts.float(), emb.float(), self.ids.long(), label.float()

    def __len__(self):
        if self.label_type == '10day_avg':
            return self.timeseries.shape[1] - self.context_window - 10
        elif self.label_type == '5day_avg':
            return self.timeseries.shape[1] - self.context_window - 5
        else:
            return self.timeseries.shape[1] - self.context_window - 1
