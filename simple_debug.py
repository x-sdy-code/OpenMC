import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class SourcePosition:
    position: Tuple[float, float, float]
    is_occupied: bool = False
    current_source: Optional[int] = None

def analyze_source_rack():
    """分析源架的结构"""
    
    # 参数
    y_length = 1.0  
    z_length = 2.0
    nlayer = 4
    sources_per_layer = 50
    
    print("=== 源架参数 ===")
    print(f"Y长度: {y_length}m")
    print(f"Z长度: {z_length}m") 
    print(f"层数: {nlayer}")
    print(f"每层源数: {sources_per_layer}")
    
    # 关键参数计算
    max_source_length = 0.408
    available_z_length = z_length - max_source_length  # 1.592
    layer_spacing = available_z_length / (nlayer - 1)  # 1.592/3 = 0.531
    
    sources_per_quadrant = sources_per_layer // 2  # 25
    source_diameter = 0.02
    
    # Y方向间距计算
    available_y_length_quadrant = (y_length / 2) - source_diameter  # 0.48
    source_spacing_quadrant = available_y_length_quadrant / (sources_per_quadrant - 1)  # 0.48/24 = 0.02
    
    print(f"\n=== 计算结果 ===")
    print(f"可用Z长度: {available_z_length:.3f}m")
    print(f"层间距: {layer_spacing:.3f}m")
    print(f"每象限源数: {sources_per_quadrant}")
    print(f"每象限可用Y长度: {available_y_length_quadrant:.3f}m")
    print(f"象限内源间距: {source_spacing_quadrant:.3f}m")
    
    # Z坐标计算
    z_start = -available_z_length / 2  # -0.796
    print(f"\nZ坐标计算:")
    print(f"Z起始: {z_start:.3f}m")
    
    for layer in range(nlayer):
        z = z_start + layer * layer_spacing
        print(f"层{layer+1}: Z = {z:.3f}m")
    
    # Y坐标计算（以右上象限为例）
    print(f"\nY坐标计算（右上象限）:")
    y_start_quadrant = 0.0
    y_center = y_length / 4  # 0.25
    
    for i in range(min(5, sources_per_quadrant)):  # 只打印前5个
        y_quadrant = y_start_quadrant + i * source_spacing_quadrant
        y = y_quadrant + y_center
        print(f"源{i+1}: Y象限内位置={y_quadrant:.3f}, 实际Y={y:.3f}")
    
    # 分析四分之一区域
    print(f"\n=== 四分之一区域分析 ===")
    half_layers = nlayer // 2  # 2
    print(f"上半层: 层{half_layers+1}到层{nlayer}")
    
    # 模拟位置索引
    positions_created = 0
    for layer in range(nlayer):
        layer_start = positions_created
        
        # 每层创建positions
        for i in range(sources_per_quadrant):
            for quadrant in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                positions_created += 1
        
        layer_end = positions_created - 1
        print(f"层{layer+1}: 位置索引 {layer_start} - {layer_end} (共{layer_end-layer_start+1}个)")
        
        # 检查上半层
        if layer >= half_layers:
            z = z_start + layer * layer_spacing
            print(f"  -> 上半层，Z={z:.3f}")
            
            # 计算右上象限位置数
            right_upper_count = 0
            for i in range(sources_per_quadrant):
                y_quadrant = y_start_quadrant + i * source_spacing_quadrant
                y = y_quadrant + y_center  # 右上象限：Y>0
                if y > 0 and z > 0:
                    right_upper_count += 1
            
            print(f"     右上象限(y>0,z>0)位置数: {right_upper_count}")
    
    print(f"\n总位置数: {positions_created}")

if __name__ == "__main__":
    analyze_source_rack()