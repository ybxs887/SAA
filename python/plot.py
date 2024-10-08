import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei以显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
# 设置全局字体大小
plt.rcParams.update({'font.size': 15})  # 设置全局字体大小
# 数据
data_size = np.array([8, 32, 73, 130, 204, 294, 400, 524, 663])
mbs_150mhz = np.array([0.73, 0.8800, 1.130, 1.430, 1.47, 1.50, 1.51, 1.52200, 1.53])

# 绘制散点图
plt.figure(figsize=(10, 6))
x_ticks = np.linspace(min(data_size), max(data_size), num=len(data_size))
plt.scatter(x_ticks, mbs_150mhz, label='200MHz', color='blue')

# 将散点连成线
plt.plot(x_ticks, mbs_150mhz, color='blue', linestyle='-', linewidth=2)

# 添加标题和标签，同时设置字体大小
plt.xlabel('数据量(Kb)')  # 设置X轴标题并设置字体大小
plt.ylabel('传输速率(Gbps)')  # 设置Y轴标题并设置字体大小


# 添加网格线
plt.grid(True)

# 设置 x 轴和 y 轴的范围
plt.ylim(0, 2)

# 添加自定义 x 轴刻度
# 使用 linspace 创建均匀分布的 x 轴刻度，然后映射到 data_size 的值
plt.xticks(x_ticks, data_size, rotation=45)  # 设置刻度字体大小

# # 显示图表
# plt.show()

# 保存图片，设置 dpi 和字体大小
plt.savefig('plot.png', dpi=300, bbox_inches='tight')  # 保存图片，提高图片分辨率，并确保文字不被裁剪
