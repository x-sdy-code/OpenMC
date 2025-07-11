import numpy as np
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import random
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.problem import ElementwiseProblem
from pymoo.visualization.scatter import Scatter
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from multiprocessing import Pool  # 添加这行！导入Pool类

# 设置matplotlib支持中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

@dataclass
class SourcePosition:
    """表示源架上的一个固定位置"""
    position: Tuple[float, float, float]  # 位置坐标(x,y,z)
    is_occupied: bool = False             # 是否被占用
    current_source: Optional[int] = None  # 当前放置的源ID(如果有)

@dataclass
class SourceGroup:
    """表示一组具有相同活度的源"""
    activity: float              # 活度(Bq)
    length: float = 0.408        # 线源长度(m)
    count: int = 0               # 该组源的数量
    positions: List[int] = field(default_factory=list)  # 分配给该组的位置ID列表

@dataclass
class Grid:
    """表示计算剂量的三维网格"""
    x_range: Tuple[float, float]  # X轴范围 (min, max)
    y_range: Tuple[float, float]  # Y轴范围 (min, max)
    z_range: Tuple[float, float]  # Z轴范围 (min, max)
    bins: Tuple[int, int, int]    # 各轴分箱数 (x_bins, y_bins, z_bins)
    
    def generate_points(self) -> np.ndarray:
        """生成网格中心点的坐标"""
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        z_min, z_max = self.z_range
        x_bins, y_bins, z_bins = self.bins
        
        # 计算步长
        x_step = (x_max - x_min) / x_bins
        y_step = (y_max - y_min) / y_bins
        z_step = (z_max - z_min) / z_bins
        
        # 生成网格中心点坐标
        x = np.linspace(x_min + x_step / 2, x_max - x_step / 2, x_bins)
        y = np.linspace(y_min + y_step / 2, y_max - y_step / 2, y_bins)
        z = np.linspace(z_min + z_step / 2, z_max - z_step / 2, z_bins)
        
        # 保存坐标向量供外部使用
        self.x_coords = x
        self.y_coords = y
        self.z_coords = z
        
        # 使用meshgrid生成所有坐标组合
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        
        # 将坐标展平为Nx3数组
        return np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T

class DoseCalculator:
    """计算线源辐射剂量的计算器"""
    def __init__(self, physical_params: dict = None):
        # 物理常数
        self.Tk = 8.67E-17  # 比释动能率常数, Gy m2 Bq−1 s−1
        self.upw_upair = 37.64 / 33.85  # 水与空气的比释动能转换因子
        
        # 可选的物理参数覆盖
        if physical_params:
            self.__dict__.update(physical_params)
    
    def calculate_dose_at_point(self, point: Tuple[float, float, float], 
                               source_positions: List[SourcePosition],
                               source_groups: Dict[int, SourceGroup]) -> float:
        """计算单个点相对于所有源的总剂量"""
        px, py, pz = point
        total_dose = 0.0
        
        for pos in source_positions:
            if not pos.is_occupied:
                continue
                
            group_id = pos.current_source
            if group_id is None:
                continue
                
            group = source_groups[group_id]
            sx, sy, sz = pos.position
            
            # 线源两端点坐标
            source_end_1 = [sx, sy, sz - group.length / 2]
            source_end_2 = [sx, sy, sz + group.length / 2]
            
            # 计算场点到线源的垂直距离
            h = math.hypot(px - sx, py - sy)
            
            if h == 0:
                # 场点在线源轴上，使用特殊处理
                z_diff_1 = pz - (sz - group.length / 2)
                z_diff_2 = pz - (sz + group.length / 2)
                
                if z_diff_1 * z_diff_2 > 0:  # 场点在线源外部
                    R1 = math.hypot(h, z_diff_1)
                    R2 = math.hypot(h, z_diff_2)
                    if h > 0:
                        dose_rate = (group.activity * self.Tk) / (h * group.length) * math.log(R2 / R1) * self.upw_upair
                    else:
                        dose_rate = 0.0
                else:  # 场点在线源内部
                    z_center = sz
                    R_center = math.hypot(h, pz - z_center)
                    if R_center > 0:
                        dose_rate = (group.activity * self.Tk) / (R_center ** 2) * self.upw_upair
                    else:
                        dose_rate = 0.0
            else:
                # 计算场点与线源两端点的张角
                theta1 = self._calculate_angle(point, source_end_1)
                theta2 = self._calculate_angle(point, source_end_2)
                
                # 计算张角差
                delta_theta = abs(theta2 - theta1)
                
                # 计算剂量率
                if h > 0 and group.length > 0:
                    exposure_rate = (group.activity * self.Tk) / (h * group.length) * delta_theta
                    dose_rate = exposure_rate * self.upw_upair
                else:
                    dose_rate = 0.0
            
            total_dose += dose_rate
        
        return total_dose
    
    def calculate_dose_distribution(self, grid: Grid, 
                                   source_positions: List[SourcePosition],
                                   source_groups: Dict[int, SourceGroup]) -> np.ndarray:
        """计算整个网格上的剂量分布"""
        # 生成计算点
        points = grid.generate_points()
        num_points = points.shape[0]
        
        # 初始化剂量数组
        dose_values = np.zeros(num_points)
        
        # 对每个点计算总剂量
        for i, point in enumerate(points):
            dose_values[i] = self.calculate_dose_at_point(
                point, source_positions, source_groups)
        
        # 重塑为三维数组
        x_bins, y_bins, z_bins = grid.bins
        return dose_values.reshape(x_bins, y_bins, z_bins)
    
    def _calculate_angle(self, p: Tuple[float, float, float], 
                        end: Tuple[float, float, float]) -> float:
        """计算场点与线源端点的张角"""
        # 向量：从场点到线源端点
        vec = [end[i] - p[i] for i in range(3)]
        # Z轴线源方向向量
        line_dir = [0, 0, 1]
        
        # 计算叉积模长
        cross_x = vec[1] * line_dir[2] - vec[2] * line_dir[1]
        cross_y = vec[2] * line_dir[0] - vec[0] * line_dir[2]
        cross_z = vec[0] * line_dir[1] - vec[1] * line_dir[0]
        cross_mag = math.hypot(cross_x, cross_y, cross_z)
        
        # 计算点积
        dot = sum(vec[i] * line_dir[i] for i in range(3))
        
        # 计算角度
        vec_mag = math.hypot(*vec)
        if vec_mag > 0:
            sin_theta = cross_mag / (vec_mag * math.hypot(*line_dir))
            # 限制在[-1,1]防止数值误差
            sin_theta = max(min(sin_theta, 1.0), -1.0)
            theta = math.asin(sin_theta)
            
            # 根据点积判断角度方向
            if dot < 0:
                theta = math.pi - theta
            return theta
        else:
            return 0.0

def plot_slice(plane, index, data, grid, title=None):
    """
    在指定平面上绘制切片图
    
    参数:
    plane: 切片平面，可选 'XY', 'YZ', 'XZ'
    index: 切片位置索引
    data: 三维数据数组
    grid: Grid对象，包含坐标信息
    title: 图表标题，默认为None
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if plane == 'XY':
        # XY平面切片（固定z）
        slice_data = data[:, :, index]
        im = ax.imshow(slice_data.T, origin='lower', 
                      extent=[grid.x_range[0], grid.x_range[1], 
                             grid.y_range[0], grid.y_range[1]],
                      aspect='auto', cmap='viridis')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        if title is None:
            title = f'XY (Z = {grid.z_coords[index]:.3f} m)'
            
    elif plane == 'YZ':
        # YZ平面切片（固定x）
        slice_data = data[index, :, :]
        im = ax.imshow(slice_data.T, origin='lower', 
                      extent=[grid.y_range[0], grid.y_range[1], 
                             grid.z_range[0], grid.z_range[1]],
                      aspect='auto', cmap='viridis')
        ax.set_xlabel('Y [m]')
        ax.set_ylabel('Z [m]')
        if title is None:
            title = f'YZ (X = {grid.x_coords[index]:.3f} m)'
            
    elif plane == 'XZ':
        # XZ平面切片（固定y）
        slice_data = data[:, index, :]
        im = ax.imshow(slice_data.T, origin='lower', 
                      extent=[grid.x_range[0], grid.x_range[1], 
                             grid.z_range[0], grid.z_range[1]],
                      aspect='auto', cmap='viridis')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Z [m]')
        if title is None:
            title = f'XZ (Y = {grid.y_coords[index]:.3f} m)'
    
    else:
        raise ValueError("无效的平面参数。请选择 'XY', 'YZ' 或 'XZ'")
    
    fig.colorbar(im, ax=ax, label=f'Gy/s')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f'dose_slice_{plane}_{index}.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def create_source_rack_grid_random(y_length: float, z_length: float, ny: int, nz: int,
                                   source_groups: Dict[int, SourceGroup]) -> List[SourcePosition]:
    """
    创建YZ平面上的单板型源架的源位置网格，采用随机分布方式
    
    参数:
    y_length: Y方向的长度(m)
    z_length: Z方向的长度(m)
    ny: Y方向的位置数量
    nz: Z方向的位置数量
    source_groups: 源组字典，用于获取源的实际长度
    
    返回:
    源位置列表
    """
    positions = []
    
    # 获取所有源组中最大的源长度
    max_source_length = max(group.length for group in source_groups.values())
    
    # 在YZ平面上创建均匀分布的位置网格
    y_coords = np.linspace(-y_length/2, y_length/2, ny)
    z_coords = np.linspace(-z_length/2, z_length/2, nz)
    
    # 为每个网格点创建一个源位置
    for z in z_coords:
        for y in y_coords:
            x = 0.0  # X坐标始终为0，因为源架在YZ平面
            positions.append(SourcePosition(position=(x, y, z)))
    
    print(f"创建了 {len(positions)} 个源位置 ({ny} × {nz} 网格)")
    print(f"Y方向范围: {-y_length/2:.3f} 到 {y_length/2:.3f} m")
    print(f"Z方向范围: {-z_length/2:.3f} 到 {z_length/2:.3f} m")
    
    return positions

def plot_source_positions(source_positions: List[SourcePosition], 
                         source_groups: Dict[int, SourceGroup],
                         y_length: float, z_length: float):
    """
    绘制源架上的源位置示意图
    
    参数:
    source_positions: 源位置列表
    source_groups: 源组字典
    y_length: Y方向的长度(m)
    z_length: Z方向的长度(m)
    """
    # 统计每种源类型的数量
    print("\n===== 源分配统计 =====")
    occupied_count = sum(1 for pos in source_positions if pos.is_occupied)
    print(f"总位置数: {len(source_positions)}")
    print(f"已占用位置: {occupied_count}")
    
    for group_id, group in source_groups.items():
        count = sum(1 for pos in source_positions if pos.is_occupied and pos.current_source == group_id)
        print(f"源组 {group_id}: {count} 个源")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制背景网格线
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 设置坐标轴范围
    ax.set_xlim(-y_length/2*1.1, y_length/2*1.1)
    ax.set_ylim(-z_length/2*1.1, z_length/2*1.1)
    
    # 绘制源架边界
    y_min, y_max = -y_length/2, y_length/2
    z_min, z_max = -z_length/2, z_length/2
    ax.plot([y_min, y_max], [z_max, z_max], 'k--', alpha=0.5, linewidth=2)  # 顶部边界
    ax.plot([y_min, y_max], [z_min, z_min], 'k--', alpha=0.5, linewidth=2)  # 底部边界
    ax.plot([y_min, y_min], [z_min, z_max], 'k--', alpha=0.5, linewidth=2)  # 左侧边界
    ax.plot([y_max, y_max], [z_min, z_max], 'k--', alpha=0.5, linewidth=2)  # 右侧边界

    # 设置坐标轴标签和标题
    ax.set_xlabel('Y [m]')
    ax.set_ylabel('Z [m]')
    ax.set_title('源架上的随机源分布示意图')
    
    # 用于存储每个源组的第一个矩形，以便创建图例
    legend_elements = []
    
    # 首先绘制所有空位置
    for pos in source_positions:
        if not pos.is_occupied:
            x, y, z = pos.position
            ax.plot(y, z, 'o', color='lightgray', markersize=3, alpha=0.5)
    
    # 然后绘制占用的位置
    colors = plt.cm.Set1(np.linspace(0, 1, len(source_groups)))
    
    for pos in source_positions:
        if not pos.is_occupied:
            continue
            
        group_id = pos.current_source
        group = source_groups[group_id]
        x, y, z = pos.position
        
        # 线源的实际长度
        line_length = group.length
        
        # 绘制线源（用圆点表示，大小与活度相关）
        marker_size = 50 + group.activity * 1e-14  # 根据活度调整大小
        marker_size = min(marker_size, 200)  # 限制最大大小
        
        scatter = ax.scatter(y, z, s=marker_size, c=[colors[group_id]], 
                           alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # 仅为每个源组添加一个图例项
        if group_id not in [item[0] for item in legend_elements]:
            legend_elements.append((group_id, scatter, f'组 {group_id}: {group.activity:.1e} Bq'))
    
    # 添加图例
    if legend_elements:
        # 提取图例元素和标签
        handles = [item[1] for item in legend_elements]
        labels = [item[2] for item in legend_elements]
        
        # 创建图例
        ax.legend(handles, labels, loc='upper right')
    
    # 添加说明文字
    ax.text(0.02, 0.98, f'总位置: {len(source_positions)}\n已占用: {occupied_count}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('source_distribution_random.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

class SourceOptimizationProblemRandom(ElementwiseProblem):
    """完全随机源分布优化问题"""
    
    def __init__(self, source_positions: List[SourcePosition], 
                source_groups: Dict[int, SourceGroup],
                grid: Grid):
        """
        初始化完全随机源分布优化问题
        
        参数:
        source_positions: 源架上的固定位置列表
        source_groups: 源组字典
        grid: 计算剂量的网格
        """
        # 计算总源数量
        total_sources = sum(group.count for group in source_groups.values())
        
        # 决策变量是一个排列，表示源在位置上的分配
        # 长度为总源数量，每个值表示选择的位置索引
        super().__init__(n_var=total_sources, 
                         n_obj=2,  # 两个目标：最小化DUR和最大化最小剂量
                         n_ieq_constr=0,
                         xl=0, xu=len(source_positions)-1,
                         vtype=int)
        
        self.source_positions = source_positions
        self.source_groups = source_groups
        self.grid = grid
        self.dose_calculator = DoseCalculator()
        
        # 创建源类型列表（根据源组的数量和顺序）
        self.source_types = []
        for group_id, group in source_groups.items():
            self.source_types.extend([group_id] * group.count)
        
        print(f"初始化优化问题: {total_sources} 个源, {len(source_positions)} 个位置")
        print(f"源类型分布: {[self.source_types.count(gid) for gid in source_groups.keys()]}")
    
    def _evaluate(self, x, out, *args, **kwargs):
        """评估函数，计算目标函数值"""
        # 重置所有源位置状态
        for pos in self.source_positions:
            pos.is_occupied = False
            pos.current_source = None
        
        # 完全随机分配：直接使用决策变量x作为位置索引
        # x是一个长度为total_sources的数组，每个元素是选择的位置索引
        
        # 确保位置索引不重复
        unique_positions = []
        for pos_idx in x:
            if pos_idx not in unique_positions and pos_idx < len(self.source_positions):
                unique_positions.append(pos_idx)
        
        # 如果唯一位置不够，随机选择剩余位置
        if len(unique_positions) < len(x):
            available_positions = [i for i in range(len(self.source_positions)) 
                                 if i not in unique_positions]
            needed = len(x) - len(unique_positions)
            if len(available_positions) >= needed:
                additional = random.sample(available_positions, needed)
                unique_positions.extend(additional)
        
        # 分配源到选定的位置
        for i, pos_idx in enumerate(unique_positions[:len(self.source_types)]):
            self.source_positions[pos_idx].is_occupied = True
            self.source_positions[pos_idx].current_source = self.source_types[i]
        
        # 计算剂量分布
        dose_distribution = self.dose_calculator.calculate_dose_distribution(
            self.grid, self.source_positions, self.source_groups)
        
        # 计算目标函数
        # 目标1：最小化DUR (剂量不均匀性比)
        min_dose = np.min(dose_distribution)
        max_dose = np.max(dose_distribution)
        dur = max_dose / min_dose if min_dose > 0 else float('inf')
        
        # 目标2：最大化1m处YZ平面上的最小剂量（取负号变为最小化问题）
        min_dose_1m = -min_dose  # 取负号，因为pymoo默认最小化
        
        # 设置目标函数值
        out["F"] = [dur, min_dose_1m]

def main():
    print("===== 完全随机源分布优化 =====")
    
    # 创建源架上的固定位置（单板型，YZ平面）
    y_length = 1  # Y方向长度(m)
    z_length = 2  # Z方向长度(m)
    ny = 20       # Y方向位置数量
    nz = 24       # Z方向位置数量
    total_positions = ny * nz  # 总位置数 = 480
    
    # 定义2组不同活度的源，共120个源（远少于总位置数，允许更多随机性）
    activityi = 10000 * 3.7E10
    source_groups = {
        0: SourceGroup(activity=activityi, count=80),      # 第1组：高活度源
        1: SourceGroup(activity=0.8E14, count=40),        # 第2组：中等活度源
    }
    total_sources = sum(group.count for group in source_groups.values())
    
    print(f"\n源架配置：{ny} × {nz} = {total_positions} 个位置")
    print(f"源配置：共{len(source_groups)}组，总源数量={total_sources}")
    print(f"位置利用率：{total_sources/total_positions:.1%}")
    
    # 定义计算网格，在1m处的YZ平面
    grid = Grid(
        x_range=(0.95, 1.05),  # 在1m附近
        y_range=(-y_length/2, y_length/2),
        z_range=(-z_length/2, z_length/2),
        bins=(1, 50, 50)  # 在YZ平面上有足够的分辨率
    )

    # 创建随机分布的源架位置
    source_positions = create_source_rack_grid_random(y_length, z_length, ny, nz, source_groups)
    
    # 创建完全随机的优化问题
    problem = SourceOptimizationProblemRandom(source_positions, source_groups, grid)
    
    # 创建参考方向（用于NSGA-III）
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)
    
    # 配置NSGA-III算法
    algorithm = NSGA3(
        pop_size=50,  # 增加种群大小以探索更多随机配置
        ref_dirs=ref_dirs,
        sampling=PermutationRandomSampling(),
        crossover=PointCrossover(n_points=2),
        mutation=InversionMutation(prob=0.15),  # 增加变异概率以增强随机性
        eliminate_duplicates=True
    )
    
    print("\n开始优化（完全随机源分布）...")
    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', 5),  # 适当的代数
        verbose=True,
        save_history=True,
    )
    
    # 输出优化结果
    print("\n优化完成!")
    print(f"找到 {len(res.F)} 个帕累托最优解")
    
    # 可视化帕累托前沿
    plot = Scatter(title="帕累托前沿 (随机分布)")
    plot.add(res.F, color="red", marker="o")
    plot.show()
    
    # 分析最佳解
    best_idx = np.argmin(res.F[:, 0])  # 选择DUR最小的解
    best_x = res.X[best_idx]
    
    # 重置所有源位置状态
    for pos in source_positions:
        pos.is_occupied = False
        pos.current_source = None
    
    # 根据最佳解分配源到位置
    problem._evaluate(best_x, {})  # 复用评估函数中的分配逻辑
    
    # 计算并显示最佳解的剂量分布
    print("\n最佳解的剂量分布:")
    dose_distribution = problem.dose_calculator.calculate_dose_distribution(
        grid, source_positions, source_groups)
    
    # 输出剂量统计信息
    print(f"剂量分布 - 最小值: {np.min(dose_distribution):.2e} Gy/s")
    print(f"剂量分布 - 最大值: {np.max(dose_distribution):.2e} Gy/s")
    print(f"剂量分布 - DUR: {np.max(dose_distribution)/np.min(dose_distribution):.2f}")
    print(f"剂量分布 - 平均值: {np.mean(dose_distribution):.2e} Gy/s")
    print(f"剂量分布 - 标准差: {np.std(dose_distribution):.2e} Gy/s")
    print(f"剂量分布 - 变异系数: {np.std(dose_distribution)/np.mean(dose_distribution):.2%}")
    
    # 绘制最佳解的随机源分布
    plot_source_positions(source_positions, source_groups, y_length, z_length)
    
    # 绘制YZ平面的剂量分布
    plot_slice('YZ', 0, dose_distribution, grid, "优化后的YZ平面剂量分布 (随机源分布)")

if __name__ == "__main__":
    main()