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
from pymoo.decomposition.asf import ASF
from pymoo.util.misc import stack

# 设置matplotlib支持中文显示（保留中文注释）
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
            title = f'XY Plane (Z = {grid.z_coords[index]:.3f} m)'
            
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
            title = f'YZ Plane (X = {grid.x_coords[index]:.3f} m)'
            
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
            title = f'XZ Plane (Y = {grid.y_coords[index]:.3f} m)'
    
    else:
        raise ValueError("Invalid plane parameter. Please choose 'XY', 'YZ' or 'XZ'")
    
    fig.colorbar(im, ax=ax, label='Dose Rate [Gy/s]')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    plt.close()


def create_source_rack_grid(y_length: float, z_length: float, nlayer: int, sources_per_layer: int,
                            source_groups: Dict[int, SourceGroup]) -> List[SourcePosition]:
    """
    创建YZ平面上的源架的源位置网格，考虑源的长度避免重叠，并在四个象限中镜像分布
    
    参数:
    y_length: Y方向的长度(m)
    z_length: Z方向的长度(m)
    nlayer: Z方向的层数
    sources_per_layer: 每层源的数量（单象限）
    source_groups: 源组字典，用于获取源的实际长度
    
    返回:
    源位置列表
    """
    positions = []
    
    # 获取所有源组中最大的源长度
    max_source_length = max(group.length for group in source_groups.values())
    
    # 确保nlayer是偶数
    if nlayer % 2 != 0:
        raise ValueError("nlayer must be even")
    
    # 计算层间距，确保层间距离至少为源长度的1.2倍，避免层间源重叠
    min_layer_spacing = max_source_length * 1.2
    
    # 计算可用的Z范围，考虑到最高层顶面与Zmax重合，最底层底面与Zmin重合
    available_z_length = z_length - max_source_length
    
    # 计算实际层间距
    layer_spacing = available_z_length / (nlayer - 1) if nlayer > 1 else 0
    
    # 计算每层源的间距，确保源间距离至少为源直径的1.2倍（假设源直径为0.02m）
    source_diameter = 0.02  # 假设源的直径为0.02m
    min_source_spacing = source_diameter * 1.2
    
    # 计算单个象限的源间距和起始位置
    # 这里将每层源数量除以2，因为我们要在两个象限中分布
    sources_per_quadrant = sources_per_layer // 2
    
    if sources_per_quadrant < 1:
        raise ValueError("sources_per_layer must be at least 2 to distribute in multiple quadrants")
    
    # 计算单个象限的源间距
    available_y_length_quadrant = (y_length / 2) - source_diameter
    source_spacing_quadrant = available_y_length_quadrant / (sources_per_quadrant - 1) if sources_per_quadrant > 1 else 0
    source_spacing_quadrant = max(source_spacing_quadrant, min_source_spacing)
    
    # 计算Y方向的起始位置（相对于象限中心）
    y_start_quadrant = 0.0  # 从象限中心开始
    
    # 计算Z方向的起始位置（考虑对称性）
    z_start = -available_z_length / 2
    
    # 生成源位置
    for layer in range(nlayer):
        # 计算当前层的Z坐标
        z = z_start + layer * layer_spacing
        
        print(f"Layer {layer+1} Z-coordinate: {z:.4f}, Source length range: Bottom {z - max_source_length/2:.4f} to Top {z + max_source_length/2:.4f}")
        
        for i in range(sources_per_quadrant):
            # 计算Y坐标（相对于象限中心）
            y_quadrant = y_start_quadrant + i * source_spacing_quadrant
            
            # 为四个象限创建源位置
            for quadrant in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                quadrant_sign_y, quadrant_sign_z = quadrant
                
                # 计算实际Y坐标（考虑象限符号）
                y = quadrant_sign_y * (y_quadrant + (y_length / 4))  # 调整到象限中心
                
                # 确保源不会超出边界
                y_left = y - source_diameter/2
                y_right = y + source_diameter/2
                
                if not (-y_length/2 <= y_left and y_right <= y_length/2):
                    # 源超出Y边界，调整位置
                    y_center = (y_left + y_right) / 2
                    if y_center > 0:
                        y = (y_length/2 - source_diameter/2) * quadrant_sign_y
                    else:
                        y = (-y_length/2 + source_diameter/2) * quadrant_sign_y
                    
                    y_left = y - source_diameter/2
                    y_right = y + source_diameter/2
                
                # X坐标始终为0，因为源架在YZ平面
                x = 0.0
                
                # 打印每个源的Y坐标和边界
                if i == sources_per_quadrant - 1 and quadrant == (1, 1):  # 只打印第一象限的最后一个源
                    print(f"  Source {i+1} (Quadrant {quadrant}): Y-coordinate: {y:.4f}, Left-Right boundary: {y_left:.4f} to {y_right:.4f}, Deviation from Ymax: {y_length/2 - y_right:.6f}")
                
                positions.append(SourcePosition(position=(x, y, z)))
    
    print(f"Created {len(positions)} source positions")
    return positions

def plot_source_positions(source_positions: List[SourcePosition], 
                         source_groups: Dict[int, SourceGroup],
                         y_length: float, z_length: float,
                         nlayer: int, sources_per_layer: int):
    """
    绘制源架上的源位置示意图
    
    参数:
    source_positions: 源位置列表
    source_groups: 源组字典
    y_length: Y方向的长度(m)
    z_length: Z方向的长度(m)
    nlayer: 层数
    sources_per_layer: 每层源数量
    """
    # 统计每层源数量
    print("\n===== Source Allocation Statistics =====")
    layer_counts = [0] * nlayer
    layer_high_counts = [0] * nlayer
    layer_medium_counts = [0] * nlayer
    
    for i in range(nlayer):
        # 计算该层的起始和结束索引
        start = i * sources_per_layer * 4  # 每层的位置数是sources_per_layer * 4（四个象限）
        end = start + sources_per_layer * 4
        
        # 确保索引不超出范围
        if start >= len(source_positions):
            print(f"Layer {i+1}: No positions available (index out of range)")
            continue
            
        layer_pos = source_positions[start:end]
        
        # 统计每层总源数和不同类型源的数量
        count = sum(1 for pos in layer_pos if pos.is_occupied)
        high_count = sum(1 for pos in layer_pos if pos.is_occupied and pos.current_source == 0)
        medium_count = sum(1 for pos in layer_pos if pos.is_occupied and pos.current_source == 1)
        
        layer_counts[i] = count
        layer_high_counts[i] = high_count
        layer_medium_counts[i] = medium_count
        
        print(f"Layer {i+1}: Total Sources={count}, High Activity={high_count}, Medium Activity={medium_count}")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制背景网格线
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 设置坐标轴范围（严格限制在源架范围内）
    ax.set_xlim(-y_length/2*1.1, y_length/2*1.1)
    ax.set_ylim(-z_length/2*1.1, z_length/2*1.5)
    
    # 绘制源架边界 - 仅在Y方向边界上绘制虚线
    y_min, y_max = -y_length/2, y_length/2
    z_min, z_max = -z_length/2, z_length/2
    ax.plot([y_min, y_max], [z_max, z_max], 'k--', alpha=0.5)  # 顶部边界
    ax.plot([y_min, y_max], [z_min, z_min], 'k--', alpha=0.5)  # 底部边界
    ax.plot([y_min, y_min], [z_min, z_max], 'k--', alpha=0.5)  # 左侧边界
    ax.plot([y_max, y_max], [z_min, z_max], 'k--', alpha=0.5)  # 右侧边界

    # 设置坐标轴标签和标题
    ax.set_xlabel('Y [m]')
    ax.set_ylabel('Z [m]')
    ax.set_title('Source Distribution on the Rack')
    
    # 用于存储每个源组的第一个矩形，以便创建图例
    legend_elements = []
    
    # 绘制所有源
    colors = plt.cm.plasma(np.linspace(0, 1, len(source_groups)))
    
    for pos in source_positions:
        if not pos.is_occupied:
            continue
            
        group_id = pos.current_source
        group = source_groups[group_id]
        x, y, z = pos.position
        
        # 跳过源架范围外的源（防止标签超出范围）
        if not (-y_length/2 <= y <= y_length/2 and -z_length/2 <= z <= z_length/2):
            continue
            
        # 线源的实际长度
        line_length = group.length
        
        # 绘制线源（用细长方形表示）
        rect_width = 0.01  # 线源的宽度，为了可视化效果
        rect = plt.Rectangle(
            (y - rect_width/2, z - line_length/2),  # 左下角坐标
            rect_width, line_length,               # 宽度和高度
            color=colors[group_id], alpha=0.8,
        )
        ax.add_patch(rect)
        
        # 计算矩形的中心点
        rect_center_y = y
        rect_center_z = z
        
        # 在线源中心标注活度，旋转90度
        ax.text(rect_center_y, rect_center_z, 
                f'{group.activity:.1e} Bq', 
                ha='center', va='center', fontsize=8,
                rotation=90, bbox=dict(facecolor='white', alpha=0.8, pad=0.3))
        
        # 仅为每个源组添加一个图例项
        if group_id not in [item[0] for item in legend_elements]:
            legend_elements.append((group_id, rect, f'Group {group_id}: {group.activity:.1e} Bq'))
    
    # 添加图例
    if legend_elements:
        # 提取图例元素和标签
        handles = [item[1] for item in legend_elements]
        labels = [item[2] for item in legend_elements]
        
        # 创建图例
        ax.legend(handles, labels, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    plt.close()

class SourceOptimizationProblem(ElementwiseProblem):
    """源分布优化问题 - 8根源棒对称布置"""
    
    def __init__(self, source_positions: List[SourcePosition], 
                source_groups: Dict[int, SourceGroup],
                grid: Grid, nlayer: int, sources_per_layer: int):
        
        # 8根源棒，四分之一区域只优化2根源棒
        quarter_sources = 2
        
        super().__init__(n_var=quarter_sources, 
                         n_obj=2,  # 两个优化目标
                         n_ieq_constr=0,
                         xl=0, xu=len(source_positions)-1,
                         vtype=int)
        
        self.source_positions = source_positions
        self.source_groups = source_groups
        self.grid = grid
        self.dose_calculator = DoseCalculator()
        
        # 8根源棒的配置：4根高活度 + 4根中等活度
        # 四分之一区域需要：1根高活度 + 1根中等活度
        self.quarter_source_types = [0, 1]  # 第一根为高活度，第二根为中等活度
        
        self.nlayer = nlayer
        self.sources_per_layer = sources_per_layer
        self.positions_per_layer = sources_per_layer * 4  # 每层4个象限
        
        # 定义四分之一区域：右上象限(y>0, z>0) + 上半层
        self.quarter_positions = []
        half_layers = nlayer // 2
        
        # 修正：每层实际的位置数 = 每象限源数 * 4个象限 = (sources_per_layer//2) * 4 = sources_per_layer * 2
        sources_per_quadrant = sources_per_layer // 2  # 25
        actual_positions_per_layer = sources_per_quadrant * 4  # 25 * 4 = 100
        
        for layer in range(half_layers, nlayer):  # 上半层（层3,4对应索引2,3）
            start_idx = layer * actual_positions_per_layer
            end_idx = start_idx + actual_positions_per_layer
            
            for pos_idx in range(start_idx, end_idx):
                if pos_idx >= len(source_positions):
                    continue
                x, y, z = source_positions[pos_idx].position
                if y > 0 and z > 0:  # 右上象限
                    self.quarter_positions.append(pos_idx)
        
        print(f"\n四分之一区域配置：")
        print(f"上半层数：{half_layers}层 (层{half_layers+1}到层{nlayer})")
        print(f"象限选择：右上象限 (y>0, z>0)")
        print(f"四分之一区域总位置数：{len(self.quarter_positions)}")
        print(f"需要布置的源棒数：{quarter_sources} (1根高活度 + 1根中等活度)")
        
        if self.quarter_positions:
            print(f"位置范围示例：")
            for i in [0, len(self.quarter_positions)//2, -1]:
                x, y, z = source_positions[self.quarter_positions[i]].position
                print(f"  位置{self.quarter_positions[i]}: (y={y:.3f}, z={z:.3f})")
    
    def _evaluate(self, x, out, *args, **kwargs):
        # 重置所有源位置
        for pos in self.source_positions:
            pos.is_occupied = False
            pos.current_source = None
        
        # 四分之一区域源分配 (2根源棒)
        quarter_assignments = []
        
        if len(self.quarter_positions) < 2:
            print("错误：四分之一区域位置不足")
            out["F"] = [float('inf'), float('inf')]
            return
        
        # 将优化变量转换为实际位置索引
        for i in range(len(x)):
            # 确保位置索引在有效范围内
            pos_idx = self.quarter_positions[int(x[i]) % len(self.quarter_positions)]
            source_type = self.quarter_source_types[i]
            
            # 检查位置是否已被占用（避免两根源棒放在同一位置）
            if self.source_positions[pos_idx].is_occupied:
                # 寻找下一个可用位置
                for alt_idx in self.quarter_positions:
                    if not self.source_positions[alt_idx].is_occupied:
                        pos_idx = alt_idx
                        break
            
            self.source_positions[pos_idx].is_occupied = True
            self.source_positions[pos_idx].current_source = source_type
            quarter_assignments.append((pos_idx, source_type))
        
        print(f"\n四分之一区域源分配 (共{len(quarter_assignments)}根)：")
        for i, (pos_idx, s_type) in enumerate(quarter_assignments):
            x, y, z = self.source_positions[pos_idx].position
            activity = self.source_groups[s_type].activity
            print(f"  源棒{i+1}: 位置{pos_idx}，坐标(y={y:.3f}, z={z:.3f})，类型{s_type}，活度{activity:.1e} Bq")
        
        # 镜像生成完整的8根源棒分布
        total_sources = self._apply_mirroring(quarter_assignments)
        print(f"镜像后总源棒数：{total_sources}")
        
        # 计算剂量分布
        dose_distribution = self.dose_calculator.calculate_dose_distribution(
            self.grid, 
            self.source_positions, 
            self.source_groups
        )
        
        # 计算目标函数
        min_dose = np.min(dose_distribution)
        max_dose = np.max(dose_distribution)
        dur = max_dose / min_dose if min_dose > 0 else float('inf')
        mean_dose = np.mean(dose_distribution)
        
        print(f"剂量分布：最小={min_dose:.2e}, 最大={max_dose:.2e}, DUR={dur:.2f}, 平均={mean_dose:.2e}")
        
        out["F"] = [dur, -mean_dose]  # 最小化DUR，最大化平均剂量
    
    def _apply_mirroring(self, quarter_assignments):
        """对称镜像：2根源棒 → 8根源棒"""
        assigned_positions = set()
        total_sources = 0
        
        # 记录四分之一区域原始位置，避免被覆盖
        for pos_idx, _ in quarter_assignments:
            assigned_positions.add(pos_idx)
            total_sources += 1
        
        print(f"\n镜像过程：")
        
        for orig_pos_idx, source_type in quarter_assignments:
            # 获取原始位置信息
            x_orig, y_orig, z_orig = self.source_positions[orig_pos_idx].position
            
            # 计算原始位置的层和在层内的相对位置  
            sources_per_quadrant = self.sources_per_layer // 2
            actual_positions_per_layer = sources_per_quadrant * 4  # 100
            layer = orig_pos_idx // actual_positions_per_layer
            pos_in_layer = orig_pos_idx % actual_positions_per_layer
            
            print(f"  原始源棒：位置{orig_pos_idx}, 坐标(y={y_orig:.3f}, z={z_orig:.3f}), 层{layer+1}")
            
            # 生成3个镜像位置
            mirror_positions = []
            
            # 1. 左上象限 (y<0, z>0) - Y轴镜像
            for test_idx in range(layer * actual_positions_per_layer, 
                                (layer + 1) * actual_positions_per_layer):
                if test_idx >= len(self.source_positions):
                    continue
                x_test, y_test, z_test = self.source_positions[test_idx].position
                if (y_test < 0 and z_test > 0 and 
                    abs(abs(y_test) - abs(y_orig)) < 0.01 and 
                    abs(z_test - z_orig) < 0.01):
                    mirror_positions.append(test_idx)
                    break
            
            # 2. 右下象限 (y>0, z<0) - Z轴镜像  
            mirror_layer = self.nlayer - 1 - layer  # 对称层
            for test_idx in range(mirror_layer * actual_positions_per_layer,
                                (mirror_layer + 1) * actual_positions_per_layer):
                if test_idx >= len(self.source_positions):
                    continue
                x_test, y_test, z_test = self.source_positions[test_idx].position
                if (y_test > 0 and z_test < 0 and 
                    abs(y_test - y_orig) < 0.01 and 
                    abs(abs(z_test) - abs(z_orig)) < 0.01):
                    mirror_positions.append(test_idx)
                    break
            
            # 3. 左下象限 (y<0, z<0) - 双轴镜像
            for test_idx in range(mirror_layer * actual_positions_per_layer,
                                (mirror_layer + 1) * actual_positions_per_layer):
                if test_idx >= len(self.source_positions):
                    continue
                x_test, y_test, z_test = self.source_positions[test_idx].position
                if (y_test < 0 and z_test < 0 and 
                    abs(abs(y_test) - abs(y_orig)) < 0.01 and 
                    abs(abs(z_test) - abs(z_orig)) < 0.01):
                    mirror_positions.append(test_idx)
                    break
            
            # 放置镜像源棒
            for i, mirror_idx in enumerate(mirror_positions):
                if mirror_idx not in assigned_positions:
                    self.source_positions[mirror_idx].is_occupied = True
                    self.source_positions[mirror_idx].current_source = source_type
                    assigned_positions.add(mirror_idx)
                    total_sources += 1
                    
                    x_mir, y_mir, z_mir = self.source_positions[mirror_idx].position
                    mirror_layer_idx = mirror_idx // actual_positions_per_layer
                    print(f"    镜像{i+1}: 位置{mirror_idx}, 坐标(y={y_mir:.3f}, z={z_mir:.3f}), 层{mirror_layer_idx+1}")
        
        return total_sources

# 创建源架上的固定位置（单板型，YZ平面）
y_length = 1  # Y方向长度(m)
z_length = 2  # Z方向长度(m)
nlayer = 4    # 4层
sources_per_layer = 50  # 每层50个源，总位置数=200

# 定义2组不同活度的源 - 总共8根源棒
activityi = 10000 * 3.7E10
source_groups = {
    0: SourceGroup(activity=activityi, count=4),     # 4根高活度源棒
    1: SourceGroup(activity=0.8E14, count=4),       # 4根中等活度源棒
}
total_sources = sum(group.count for group in source_groups.values())

print(f"\n===== 源棒布置优化问题 =====")
print(f"源架配置：{nlayer}层 × 每层{sources_per_layer}个位置 = 总共{nlayer*sources_per_layer}个位置")
print(f"源棒配置：{len(source_groups)}种类型，总共{total_sources}根源棒")
print(f"高活度源棒：{source_groups[0].count}根，活度{source_groups[0].activity:.1e} Bq")
print(f"中等活度源棒：{source_groups[1].count}根，活度{source_groups[1].activity:.1e} Bq")
print(f"对称要求：上下对称 + 左右对称")
print(f"优化策略：只优化四分之一区域的2根源棒位置，其余6根通过镜像生成")

# 定义计算网格，在1m处的YZ平面
grid = Grid(
    x_range=(0.9, 1.0),  # 固定在1m处
    y_range=(-y_length/2, y_length/2),
    z_range=(-z_length/2, z_length/2),
    bins=(1, 21, 21)  # 在YZ平面上有足够的分辨率
)

# 创建源架位置
source_positions = create_source_rack_grid(y_length, z_length, nlayer, sources_per_layer, source_groups)

# 创建优化问题
problem = SourceOptimizationProblem(source_positions, source_groups, grid, nlayer, sources_per_layer)

# 创建参考方向（用于NSGA-III）
ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=2)

# 配置NSGA-III算法 - 针对小规模问题调整参数
algorithm = NSGA3(
    pop_size=10,  # 适中的种群大小
    ref_dirs=ref_dirs,
    sampling=PermutationRandomSampling(),
    crossover=PointCrossover(n_points=1),  # 简单的单点交叉
    mutation=InversionMutation(prob=0.2),  # 适中的变异概率
    eliminate_duplicates=True
)

# 运行优化
print(f"\n===== 开始优化 =====")
res = minimize(
    problem,
    algorithm,
    termination=('n_gen', 5),  # 5代优化
    verbose=True,
    save_history=True,
)

print(f"\n===== 优化结果 =====")
print(f"找到{len(res.F)}个帕累托最优解")
if len(res.F) > 0:
    best_idx = np.argmin(res.F[:, 0])  # 选择DUR最小的解
    best_solution = res.X[best_idx]
    best_objectives = res.F[best_idx]
    
    print(f"最优解：DUR = {best_objectives[0]:.3f}, 平均剂量 = {-best_objectives[1]:.2e}")
    print(f"四分之一区域位置：{best_solution}")
    
    # 应用最优解并可视化
    problem._evaluate(best_solution, {"F": []})
    
    # 绘制最终源分布
    plot_source_positions(source_positions, source_groups, y_length, z_length, nlayer, sources_per_layer)