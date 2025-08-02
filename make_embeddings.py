import os
import math
import json
import torch
from tqdm import tqdm
import numpy as np
from datetime import datetime, timedelta
from transformers import AutoModel, AutoTokenizer

# ========= 配置 ==========
raw_path = 'stocknet-dataset/tweet/raw'
save_dir = 'stocknet-dataset/embeddings'
model_name = 'google-bert/bert-base-uncased'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
start_date = datetime.strptime('2014-01-01', "%Y-%m-%d")
end_date = datetime.strptime('2016-01-01', "%Y-%m-%d")

# ========= 加载公司名 ==========
companies = sorted(os.listdir(raw_path))

# ========= 读取推文数据 ==========
delta = end_date - start_date
all_data = []
for i in range(delta.days + 1):
    data = {}
    day = str((start_date + timedelta(days=i)).date())
    for company in companies:
        company_path = os.path.join(raw_path, company)
        co_days = os.listdir(company_path)
        if day in co_days:
            with open(os.path.join(company_path, day), encoding='utf-8') as f:
                tweets = f.read().strip().split('\n')
                data[company] = [json.loads(tweet)['text'] for tweet in tweets]
        else:
            data[company] = []
    all_data.append(data)

# ========= 加载 BERT 模型 ==========
print(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert = AutoModel.from_pretrained(model_name).to(device)
bert.eval()

# ========= 编码每条推文 ==========
all_embeddings = []
with torch.no_grad():
    for day_data in tqdm(all_data, desc="Encoding tweets"):
        day_embeddings = []
        for company in companies:
            tweets = day_data[company]
            if len(tweets) > 0:
                emb_chunks = []
                for i in range(math.ceil(len(tweets) / batch_size)):
                    batch = tweets[i * batch_size:(i + 1) * batch_size]
                    inputs = tokenizer(
                        batch,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=128
                    ).to(device)

                    outputs = bert(**inputs)
                    emb = outputs.last_hidden_state.mean(dim=1).detach().cpu()
                    emb_chunks.append(emb)

                emb = torch.cat(emb_chunks, dim=0).mean(dim=0)  # 每家公司当天平均
            else:
                emb = torch.zeros(768)
            day_embeddings.append(emb)

        all_embeddings.append(torch.stack(day_embeddings))

all_embeddings = torch.stack(all_embeddings).transpose(0, 1)  # [公司数, 日期数, 768]

# ========= 保存 ==========
safe_model_name = model_name.replace("/", "-")  # 防止路径错误
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"{safe_model_name}.npy")

np.save(save_path, all_embeddings.numpy())
print(f"\n✅ Embeddings saved to: {save_path}")
