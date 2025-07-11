import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

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

def create_source_rack_grid(y_length: float, z_length: float, nlayer: int, sources_per_layer: int,
                            source_groups: Dict[int, SourceGroup]) -> List[SourcePosition]:
    """
    创建YZ平面上的源架的源位置网格，考虑源的长度避免重叠，并在四个象限中镜像分布
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
        
        print(f"Layer {layer+1} Z-coordinate: {z:.4f}")
        
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
                
                # 详细打印位置信息
                if i == 0:  # 只打印每层第一个源
                    print(f"  Quadrant {quadrant}: Y={y:.3f}, Z={z:.3f}, Position index={len(positions)}")
                
                positions.append(SourcePosition(position=(x, y, z)))
    
    print(f"Created {len(positions)} source positions")
    return positions

# 测试参数
y_length = 1  # Y方向长度(m)
z_length = 2  # Z方向长度(m)  
nlayer = 4    # 4层
sources_per_layer = 50  # 每层50个源

# 定义源组
activityi = 10000 * 3.7E10
source_groups = {
    0: SourceGroup(activity=activityi, count=4),     # 4根高活度源棒
    1: SourceGroup(activity=0.8E14, count=4),       # 4根中等活度源棒
}

print("=== 调试源位置创建 ===")
source_positions = create_source_rack_grid(y_length, z_length, nlayer, sources_per_layer, source_groups)

print("\n=== 位置统计 ===")
positions_per_layer = sources_per_layer * 4  # 每层4个象限

# 统计四分之一区域位置
print("\n四分之一区域筛选测试:")
nlayer = 4
half_layers = nlayer // 2  # 2
print(f"上半层: 层{half_layers+1}到层{nlayer} (索引{half_layers}到{nlayer-1})")

quarter_positions = []
for layer in range(half_layers, nlayer):  # 层2,3（索引2,3）
    start_idx = layer * positions_per_layer  
    end_idx = start_idx + positions_per_layer
    
    print(f"\n检查层{layer+1} (索引范围 {start_idx}-{end_idx-1}):")
    count_y_pos = 0
    count_z_pos = 0
    count_both_pos = 0
    
    for pos_idx in range(start_idx, min(end_idx, len(source_positions))):
        x, y, z = source_positions[pos_idx].position
        if y > 0:
            count_y_pos += 1
        if z > 0:
            count_z_pos += 1
        if y > 0 and z > 0:
            count_both_pos += 1
            quarter_positions.append(pos_idx)
            if len(quarter_positions) <= 5:  # 只打印前5个
                print(f"  四分之一区域位置{pos_idx}: (y={y:.3f}, z={z:.3f})")
    
    print(f"  y>0: {count_y_pos}, z>0: {count_z_pos}, y>0且z>0: {count_both_pos}")

print(f"\n四分之一区域总位置数: {len(quarter_positions)}")

# 显示各层的Z坐标分布
print("\n=== 各层Z坐标分布 ===")
for layer in range(nlayer):
    start_idx = layer * positions_per_layer
    if start_idx < len(source_positions):
        z_coord = source_positions[start_idx].position[2]
        print(f"层{layer+1} (索引{layer}): Z = {z_coord:.4f}")