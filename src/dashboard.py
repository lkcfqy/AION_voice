import visdom
import numpy as np
import time
from src.config import VISDOM_SERVER, VISDOM_PORT, VISDOM_ENV

class AIONDashboard:
    """
    使用 Visdom 的实时可视化仪表盘。
    """
    def __init__(self):
        print(f"Connecting to Visdom at {VISDOM_SERVER}:{VISDOM_PORT}...")
        self.vis = visdom.Visdom(server=VISDOM_SERVER, port=VISDOM_PORT, env=VISDOM_ENV)
        
        if not self.vis.check_connection():
            raise ConnectionError("Could not connect to Visdom server! Please run 'python -m visdom.server'")

        self._init_plots()

    def _init_plots(self):
        """初始化 4 个强制监控面板。"""
        # 清除该环境中的所有现有图表以防止重复
        self.vis.close(env=VISDOM_ENV)

        
        # 1. 环境视图 (RGB 流)
        self.win_env = self.vis.image(
            np.zeros((3, 56, 56)),
            win="win_env",
            opts=dict(title="耳蜗 (频谱图视图)", caption="原始音频频谱输入")
        )

        # 2. LSM 光栅图
        self.win_lsm = self.vis.line(
            X=np.array([0]), Y=np.array([0]),
            win="win_lsm",
            opts=dict(
                title="1. LSM 光栅图 (脉冲)",
                xlabel="时间",
                ylabel="神经元索引",
                markers=True,
                markersize=2
            )
        )

        # 3. HDC 相似度
        self.win_hdc = self.vis.line(
            X=np.array([0]), Y=np.array([0]),
            win="win_hdc",
            opts=dict(
                title="2. HDC 相似度",
                xlabel="时间",
                ylabel="余弦相似度",
                ylim=[0, 1]
            )
        )

        # 4. 能量地形图（目前用于概念/标量追踪）
        self.win_energy = self.vis.line(
            X=np.array([0]), Y=np.array([0]),
            win="win_energy",
            opts=dict(
                title="3. 能量地形图 (MHN)",
                xlabel="步数",
                ylabel="能量 (-LSE)"
            )
        )

        # 5. 生存曲线（损失 + 饥饿）
        self.win_survival = self.vis.line(
            X=np.array([[0, 0]]), Y=np.array([[0, 0]]),
            win="win_survival",
            opts=dict(
                title="4. 生存曲线",
                xlabel="步数",
                ylabel="值",
                legend=["自由能 (惊喜度)", "孤独感"]
            )
        )
        
        self.start_time = time.time()
        self.step_count = 0

    def update_env_view(self, image):
        """
        更新耳蜗视图。
        参数:
            image: 形状为 (H, W, 3) 或 (3, H, W) 的 numpy 数组。
                   Visdom 期望的形状为 (3, H, W)。
        """
        # 确保是 HWC 格式
        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
            
        # 为了可见性进行上采样（最近邻插值）
        scale = 6
        image = image.repeat(scale, axis=0).repeat(scale, axis=1)
        
        # 为 Visdom 将 HWC 转换为 CHW
        image = image.transpose(2, 0, 1)
        
        self.vis.image(image, win=self.win_env, opts=dict(title="耳蜗 (频谱图视图)"))

    def update_lsm_raster(self, active_neuron_indices, time_step=None):
        """
        更新脉冲光栅图。
        参数:
            active_neuron_indices: 在该步激活的神经元索引列表/数组。
        """
        # 实时绘图的简化：仅分散当前步的点
        # 注意：Visdom 使用 'append' 模式更新 line 对于低频来说足够高效
        t = self.step_count if time_step is None else time_step
        if len(active_neuron_indices) > 0:
            X = np.full(len(active_neuron_indices), t)
            Y = np.array(active_neuron_indices)
            self.vis.line(X=X, Y=Y, win=self.win_lsm, update='append')

    def update_hdc_similarity(self, similarity):
        self.vis.line(X=np.array([self.step_count]), Y=np.array([similarity]), 
                      win=self.win_hdc, update='append')

    def update_energy(self, energy):
        self.vis.line(X=np.array([self.step_count]), Y=np.array([energy]), 
                      win=self.win_energy, update='append')

    def update_survival(self, free_energy, loneliness):
        self.vis.line(
            X=np.array([[self.step_count, self.step_count]]), 
            Y=np.array([[free_energy, loneliness]]), 
            win="win_survival", 
            update='append'
        )
        self.step_count += 1
