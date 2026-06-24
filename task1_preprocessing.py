# -*- coding: utf-8 -*-
"""
任务1：数据预处理（10分）
公交IC卡刷卡数据分析
"""

import numpy as np
import pandas as pd

print("=" * 60)
print("任务1：数据预处理")
print("=" * 60)

# --------------------------------------------------------
# 1.1 读取数据
# --------------------------------------------------------
# 使用 pandas 读取 CSV 文件
df = pd.read_csv("ICData.csv")

# 打印数据集前5行
print("\n>>> 数据集前5行：")
print(df.head())

# 打印数据集基本信息：行数、列数、各列数据类型
print(f"\n>>> 数据集基本信息：")
print(f"  行数(记录数): {df.shape[0]}")
print(f"  列数(字段数): {df.shape[1]}")
print(f"\n>>> 各列数据类型：")
print(df.dtypes)

# --------------------------------------------------------
# 1.2 时间解析
# --------------------------------------------------------
# 将"交易时间"列从字符串转换为 pandas datetime 类型
df["交易时间"] = pd.to_datetime(df["交易时间"])

# 从交易时间中提取小时字段（整数 0~23），新增为 hour 列
# dt.hour 返回 datetime 对象的小时部分（int 类型）
df["hour"] = df["交易时间"].dt.hour

print(f"\n>>> 时间解析完成，已新增 hour 列。")
print(f"   hour 时间范围: {df['hour'].min()}:00 ~ {df['hour'].max()}:00")

# --------------------------------------------------------
# 1.3 构造衍生字段：搭乘站点数
# --------------------------------------------------------
# ride_stops = |下车站点 - 上车站点|，取绝对值即为搭乘站数
df["ride_stops"] = (df["下车站点"] - df["上车站点"]).abs()

# 统计 ride_stops == 0 的异常记录数量
# 同站上下车视为异常数据，应删除
zero_stops_count = (df["ride_stops"] == 0).sum()
print(f"\n>>> ride_stops 为 0 的异常记录数: {zero_stops_count}")

# 删除 ride_stops == 0 的异常行
df = df[df["ride_stops"] != 0].copy()
print(f"   已删除 {zero_stops_count} 行，当前剩余行数: {df.shape[0]}")

# --------------------------------------------------------
# 1.4 缺失值检查
# --------------------------------------------------------
# 打印各列缺失值数量
print(f"\n>>> 各列缺失值数量：")
print(df.isnull().sum())

# 若存在缺失值，删除对应记录
rows_before = df.shape[0]
df = df.dropna()
rows_dropped = rows_before - df.shape[0]
if rows_dropped > 0:
    print(f"\n   检测到缺失值，已删除 {rows_dropped} 行。")
else:
    print(f"\n   无缺失值，无需删除。")

print(f"\n>>> 预处理后数据集：行数 = {df.shape[0]}, 列数 = {df.shape[1]}")

# 保存预处理后的数据，供后续任务使用
df.to_csv("ICData_preprocessed.csv", index=False, encoding="utf-8-sig")
print(f"   预处理结果已保存至 ICData_preprocessed.csv")

print("\n" + "=" * 60)
print("任务1 数据预处理 - 完成！")
print("=" * 60)
