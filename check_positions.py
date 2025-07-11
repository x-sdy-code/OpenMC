import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

@dataclass
class SourcePosition:
    position: Tuple[float, float, float]
    is_occupied: bool = False
    current_source: Optional[int] = None

@dataclass 
class SourceGroup:
    activity: float
    length: float = 0.408
    count: int = 0
    positions: List[int] = field(default_factory=list)

def create_source_rack_grid_simplified(y_length: float, z_length: float, nlayer: int, sources_per_layer: int,
                            source_groups: Dict[int, SourceGroup]) -> List[SourcePosition]:
    """简化的源架创建函数，只用于调试"""
    positions = []
    
    max_source_length = max(group.length for group in source_groups.values())
    available_z_length = z_length - max_source_length
    layer_spacing = available_z_length / (nlayer - 1) if nlayer > 1 else 0
    
    source_diameter = 0.02
    sources_per_quadrant = sources_per_layer // 2  # 25
    
    available_y_length_quadrant = (y_length / 2) - source_diameter
    source_spacing_quadrant = available_y_length_quadrant / (sources_per_quadrant - 1) if sources_per_quadrant > 1 else 0
    
    y_start_quadrant = 0.0
    z_start = -available_z_length / 2
    
    print(f"参数: sources_per_quadrant={sources_per_quadrant}, source_spacing_quadrant={source_spacing_quadrant:.4f}")
    
    for layer in range(nlayer):
        z = z_start + layer * layer_spacing
        print(f"\n=== 层{layer+1}, Z={z:.3f} ===")
        
        layer_positions = []
        
        for i in range(sources_per_quadrant):
            y_quadrant = y_start_quadrant + i * source_spacing_quadrant
            
            for quadrant_idx, quadrant in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
                quadrant_sign_y, quadrant_sign_z = quadrant
                
                # 计算实际Y坐标
                y = quadrant_sign_y * (y_quadrant + (y_length / 4))
                
                # 边界检查和调整
                y_left = y - source_diameter/2
                y_right = y + source_diameter/2
                
                if not (-y_length/2 <= y_left and y_right <= y_length/2):
                    y_center = (y_left + y_right) / 2
                    if y_center > 0:
                        y = (y_length/2 - source_diameter/2) * quadrant_sign_y
                    else:
                        y = (-y_length/2 + source_diameter/2) * quadrant_sign_y
                
                x = 0.0
                pos = SourcePosition(position=(x, y, z))
                positions.append(pos)
                layer_positions.append((len(positions)-1, x, y, z, quadrant))
                
                # 详细打印前几个位置
                if i < 3:
                    print(f"  源{i+1}-象限{quadrant}: 位置{len(positions)-1}, Y={y:.3f}")
        
        # 统计本层信息
        y_values = [pos[2] for pos in layer_positions]
        unique_y = sorted(set(y_values))
        print(f"  本层位置数: {len(layer_positions)}")
        print(f"  Y坐标范围: {min(y_values):.3f} 到 {max(y_values):.3f}")
        print(f"  唯一Y值数量: {len(unique_y)}")
        if len(unique_y) <= 10:
            print(f"  唯一Y值: {[f'{y:.3f}' for y in unique_y]}")
        
        # 检查右上象限
        right_upper = [(idx, y, z, q) for idx, x, y, z, q in layer_positions if y > 0 and z > 0]
        print(f"  右上象限(y>0,z>0)数量: {len(right_upper)}")
    
    print(f"\n总位置数: {len(positions)}")
    return positions

# 测试
y_length = 1  
z_length = 2  
nlayer = 4    
sources_per_layer = 50

source_groups = {
    0: SourceGroup(activity=3.7e14, count=4),
    1: SourceGroup(activity=8.0e13, count=4),
}

print("=== 测试源位置创建 ===")
positions = create_source_rack_grid_simplified(y_length, z_length, nlayer, sources_per_layer, source_groups)

# 四分之一区域筛选测试
print("\n=== 四分之一区域筛选测试 ===")
half_layers = nlayer // 2  # 2
positions_per_layer = sources_per_layer * 4  # 200

quarter_positions = []
for layer in range(half_layers, nlayer):  # 层2,3 (索引2,3)
    start_idx = layer * positions_per_layer
    end_idx = start_idx + positions_per_layer
    
    print(f"\n检查层{layer+1} (索引{start_idx}-{end_idx-1}):")
    
    layer_quarter = []
    for pos_idx in range(start_idx, min(end_idx, len(positions))):
        x, y, z = positions[pos_idx].position
        if y > 0 and z > 0:
            quarter_positions.append(pos_idx)
            layer_quarter.append((pos_idx, y, z))
    
    print(f"  找到右上象限位置: {len(layer_quarter)}")
    if layer_quarter:
        print(f"  Y范围: {min(pos[1] for pos in layer_quarter):.3f} - {max(pos[1] for pos in layer_quarter):.3f}")

print(f"\n四分之一区域总位置数: {len(quarter_positions)}")