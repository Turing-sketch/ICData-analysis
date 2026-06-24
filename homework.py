# -*- coding: utf-8 -*-
"""
《人工智能编程语言》第3次作业 - 公交IC卡刷卡数据分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# 任务1  数据预处理（10分）
# ============================================================
print("=" * 60)
print("任务1：数据预处理")
print("=" * 60)

# 1.1 读取数据
df = pd.read_csv("ICData.csv")
print("\n>>> 数据集前5行：")
print(df.head())
print(f"\n>>> 数据集基本信息：行数={df.shape[0]}, 列数={df.shape[1]}")
print("\n>>> 各列数据类型：")
print(df.dtypes)

# 1.2 时间解析
df["交易时间"] = pd.to_datetime(df["交易时间"])
df["hour"] = df["交易时间"].dt.hour
print(f"\n>>> 时间解析完成，hour范围: {df['hour'].min()}:00 ~ {df['hour'].max()}:00")

# 1.3 构造衍生字段
df["ride_stops"] = (df["下车站点"] - df["上车站点"]).abs()
zero_stops = (df["ride_stops"] == 0).sum()
print(f"\n>>> ride_stops=0 的异常记录: {zero_stops} 条")
df = df[df["ride_stops"] != 0].copy()
print(f"   已删除 {zero_stops} 行，剩余 {df.shape[0]} 行")

# 1.4 缺失值检查
print("\n>>> 各列缺失值数量：")
print(df.isnull().sum())
before = df.shape[0]
df = df.dropna()
if df.shape[0] < before:
    print(f"   已删除 {before - df.shape[0]} 行含缺失值的记录")
else:
    print("   无缺失值，无需删除")
print(f">>> 预处理后数据集: {df.shape[0]} 行 x {df.shape[1]} 列")

# ============================================================
# 任务2  时间分布分析（20分）
# ============================================================
print("\n" + "=" * 60)
print("任务2：时间分布分析")
print("=" * 60)

# 筛选刷卡类型=0的记录（后续任务均使用此子集）
df_card = df[df["刷卡类型"] == 0].copy()
total_count = len(df_card)

# 2(a) 使用 numpy 布尔索引统计早晚时段刷卡量
# 早峰前时段：交易时间早于 07:00（hour < 7）
early_mask = df_card["hour"].values < 7
early_count = np.sum(early_mask)
# 深夜时段：交易时间晚于 22:00（hour >= 22）
late_mask = df_card["hour"].values >= 22
late_count = np.sum(late_mask)

print(f"\n>>> 全天总刷卡量（刷卡类型=0）: {total_count}")
print(f"   早峰前时段 (hour<7) 刷卡量: {early_count}, 占比: {early_count/total_count*100:.2f}%")
print(f"   深夜时段   (hour>=22) 刷卡量: {late_count}, 占比: {late_count/total_count*100:.2f}%")

# 2(b) 24小时刷卡量分布柱状图 (matplotlib)
hour_counts = df_card.groupby("hour").size()

fig, ax = plt.subplots(figsize=(12, 6))
# 定义各柱颜色：早峰前=红色，深夜=橙色，其余=蓝色
colors = []
for h in range(24):
    count = hour_counts.get(h, 0)
    if h < 7:
        colors.append("#E74C3C")   # 早峰前：红色
    elif h >= 22:
        colors.append("#F39C12")   # 深夜：橙色
    else:
        colors.append("#3498DB")   # 正常时段：蓝色

bars = ax.bar(range(24), [hour_counts.get(h, 0) for h in range(24)], color=colors)

ax.set_title("24-Hour Boarding Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Boarding Count")
ax.set_xticks(range(0, 24, 2))
ax.set_xticklabels([f"{h}:00" for h in range(0, 24, 2)])

# 图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#E74C3C", label="Pre-peak (< 7:00)"),
    Patch(facecolor="#3498DB", label="Regular (7:00-21:59)"),
    Patch(facecolor="#F39C12", label="Late-night (>= 22:00)")
]
ax.legend(handles=legend_elements, loc="upper right")

ax.grid(axis="y", alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig("hour_distribution.png", dpi=150)
print(f"\n>>> 柱状图已保存: hour_distribution.png")
print("\n" + "=" * 60)
print("任务3：线路站点分析")
print("=" * 60)

def analyze_route_stops(df, route_col='线路号', stops_col='ride_stops'):
    """
    计算各线路乘客的平均搭乘站点数及其标准差。

    Parameters
    ----------
    df : pd.DataFrame
        预处理后的数据集
    route_col : str
        线路号列名
    stops_col : str
        搭乘站点数列名

    Returns
    -------
    pd.DataFrame
        包含列：线路号、mean_stops、std_stops，按 mean_stops 降序排列
    """
    # 按线路号分组，计算搭乘站点数的均值和标准差
    result = df.groupby(route_col)[stops_col].agg(["mean", "std"]).reset_index()
    # 重命名列
    result.columns = [route_col, "mean_stops", "std_stops"]
    # 按均值降序排列
    result = result.sort_values("mean_stops", ascending=False).reset_index(drop=True)
    return result

# 调用函数（使用刷卡类型=0的子集）
route_stats = analyze_route_stops(df_card)
print("\n>>> 各线路平均搭乘站点数（前10行）：")
print(route_stats.head(10))

# seaborn 水平条形图：取均值最高的前15条线路
top15 = route_stats.head(15)

fig, ax = plt.subplots(figsize=(10, 8))
# 使用 seaborn barplot，水平方向，带标准差误差棒
sns.barplot(
    data=top15, y=top15[route_stats.columns[0]].astype(str),
    x="mean_stops", palette="Blues_d",
    errorbar=("ci", 68), capsize=0.3, ax=ax
)

# 手动添加误差棒（兼容不同 seaborn 版本）
for i, (_, row) in enumerate(top15.iterrows()):
    mean_val = row["mean_stops"]
    std_val = row["std_stops"]
    ax.errorbar(mean_val, i, xerr=std_val, capsize=0.3, color="black", linewidth=0.8)

ax.set_title("Top 15 Routes by Average Ride Stops", fontsize=14, fontweight="bold")
ax.set_xlabel("Average Ride Stops")
ax.set_ylabel("Route Number")
ax.set_xlim(0, top15["mean_stops"].max() * 1.2)
ax.grid(axis="x", alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig("route_stops.png", dpi=150)
print(f"\n>>> 条形图已保存: route_stops.png")

