import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
import math

# 设置中文字体（Mac 系统）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 第一步：生成模拟的 A/B 测试数据
# ==========================================
print("正在生成模拟的 A/B 测试数据...")
np.random.seed(42)
n_users = 10000

group_a = np.random.choice([0, 1], size=5000, p=[0.90, 0.10])
group_b = np.random.choice([0, 1], size=5000, p=[0.88, 0.12])

df = pd.DataFrame({
    'user_id': range(1, 10001),
    'group': ['control'] * 5000 + ['treatment'] * 5000,
    'converted': np.concatenate([group_a, group_b])
})

# ==========================================
# 第二步：计算基础业务指标
# ==========================================
summary = df.groupby('group')['converted'].agg(['count', 'sum', 'mean'])
summary.columns = ['总用户数', '转化人数', '转化率']
print("\n--- 基础业务指标 ---")
print(summary)

# 提取用于统计检验的变量
control = df[df['group'] == 'control']['converted']
treatment = df[df['group'] == 'treatment']['converted']

p_A = control.mean()
p_B = treatment.mean()
n_A = len(control)
n_B = len(treatment)

# ==========================================
# 第三步：可视化对比并保存图片
# ==========================================
plt.figure(figsize=(8, 5))
# 修复了警告：加入了 hue='group' 和 legend=False
sns.barplot(data=df, x='group', y='converted', errorbar=None, hue='group', palette=['#A9A9A9', '#FF4500'], legend=False)
plt.title('A/B 测试转化率对比 (灰色 vs 红色)')
plt.ylabel('转化率')
plt.xlabel('用户分组')
plt.ylim(0, 0.15)

# 保存图片（一定要写在 plt.show() 之前）
plt.savefig('ab_test_result.png', dpi=300, bbox_inches='tight')
print("\n✅ 柱状图已成功保存为 ab_test_result.png")

plt.show()

# ==========================================
# 第四步：统计学假设检验 (Z-Test 算 P-value)
# ==========================================
count = np.array([treatment.sum(), control.sum()])
nobs = np.array([n_B, n_A])

stat, pval = proportions_ztest(count, nobs, alternative='larger')

print("\n--- 统计学假设检验结果 ---")
print(f"Z-statistic (Z值): {stat:.4f}")
print(f"P-value (P值): {pval:.4f}")

# ==========================================
# 第五步：计算相对提升率 (Lift) 与 95% 置信区间 (CI)
# ==========================================
print("\n--- 业务收益评估 ---")

# 1. 相对提升率 (Relative Lift)
# 公式: (B组转化率 - A组转化率) / A组转化率
lift = (p_B - p_A) / p_A
print(f"相对提升率 (Lift): {lift:.2%}")

# 2. 计算绝对提升量及其置信区间
# 绝对差值
diff = p_B - p_A

# 标准误 (Standard Error) 公式（数学人最熟悉的二项分布方差叠加）
se_diff = math.sqrt((p_A * (1 - p_A) / n_A) + (p_B * (1 - p_B) / n_B))

# 95% 置信水平对应的 Z 临界值（双侧取 1.96）
z_critical = 1.96

ci_lower_abs = diff - z_critical * se_diff
ci_upper_abs = diff + z_critical * se_diff

print(f"绝对提升量的 95% 置信区间: [{ci_lower_abs:.2%}, {ci_upper_abs:.2%}]")

# 3. 计算相对提升率 (Lift) 的置信区间（业务方最爱看这个）
# 公式：把绝对区间除以 A 组转化率
ci_lower_rel = ci_lower_abs / p_A
ci_upper_rel = ci_upper_abs / p_A

print(f"相对提升率 (Lift) 的 95% 置信区间: [{ci_lower_rel:.2%}, {ci_upper_rel:.2%}]")

# ==========================================
# 第六步：最终业务结论
# ==========================================
alpha = 0.05
print("\n--- 最终业务决策 ---")
if pval < alpha and ci_lower_abs > 0:
    print(f"✅ 结论：P-value < {alpha}，且置信区间下限 ({ci_lower_abs:.2%}) > 0。")
    print(f"👉 我们有 95% 的把握，新版本（红色按钮）能带来至少 {ci_lower_abs:.2%} 的真实提升（最高可能提升 {ci_upper_abs:.2%}）。建议立刻全量上线！")
else:
    print("❌ 结论：P-value >= 0.05，或置信区间跨越了 0。提升可能是随机波动，不建议全量上线。")