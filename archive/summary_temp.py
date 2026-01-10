import pandas as pd

df = pd.read_csv("D:\Annual report LLM project\LLM project_20251207\data\outputs\metrics_2024.csv")

# 定义：不是 NOT FOUND 就算命中
df["hit"] = df["val"].notna() & (df["val"] != "NOT FOUND")

sanity = df[df["metric_name"].isin(["ROA", "ROE"])].copy()

sanity = sanity[sanity["hit"]]   # 只看命中的
sanity["val"] = sanity["val"].astype(float)

sanity_summary = (
    sanity.groupby("metric_name")["val"]
          .agg(["min", "median", "max"])
          .reset_index()
)

print(sanity_summary)

import matplotlib.pyplot as plt

for m in ["ROA", "ROE"]:
    vals = sanity[sanity["metric_name"] == m]["val"]
    plt.hist(vals, bins=10)
    plt.title(f"{m} distribution")
    plt.xlabel(m)
    plt.ylabel("count")
    plt.show()
