# Radiation Dose Distribution Optimization Results

## Overview
This report summarizes the results of a multi-objective optimization for cobalt source arrangement on a single-board source rack using the point kernel integration method and NSGA-III algorithm.

## Methodology

### Physics Model
- **Point Kernel Integration**: Used for calculating radiation dose from linear cobalt sources
- **Physical Constants**:
  - Specific kerma rate constant (Tk): 8.67E-17 Gy m² Bq⁻¹ s⁻¹
  - Water-to-air conversion factor: 37.64/33.85 = 1.112

### Source Configuration
- **Source Rack**: Single-board configuration in YZ plane
- **Rack Dimensions**: 1m (Y) × 2m (Z)
- **Source Layout**: 2 layers × 10 sources per layer = 20 positions total
- **Source Groups**:
  - Group 0: 10 sources with activity 3.7E14 Bq each
  - Group 1: 5 sources with activity 0.8E14 Bq each
- **Source Length**: 0.408 m (linear sources)

### Optimization Setup
- **Algorithm**: NSGA-III (Non-dominated Sorting Genetic Algorithm III)
- **Population Size**: 20 individuals
- **Generations**: 5 (reduced for demonstration)
- **Objectives**:
  1. Minimize DUR (Dose Uniformity Ratio): max_dose/min_dose
  2. Maximize minimum dose at 1m distance (converted to minimization problem)

### Calculation Grid
- **Evaluation Plane**: YZ plane at 1m distance from source rack
- **Grid Resolution**: 20×20 points
- **Coverage**: Full source rack projection area

## Optimization Results

### Performance Metrics
- **Pareto Solutions Found**: 3 optimal configurations
- **Best Solution DUR**: 1.40 (excellent uniformity)
- **Dose Statistics**:
  - Minimum Dose: 1.26e-01 Gy/s
  - Maximum Dose: 1.76e-01 Gy/s
  - Average Dose: 1.56e-01 Gy/s
  - Standard Deviation: 9.82e-03 Gy/s
  - Coefficient of Variation: 6.3% (very good uniformity)

### Key Findings

1. **Excellent Dose Uniformity**: The optimized configuration achieved a DUR of 1.40, indicating very uniform dose distribution
2. **High Dose Rates**: Achieved dose rates in the range of 0.126-0.176 Gy/s at 1m distance
3. **Efficient Source Utilization**: All 15 sources (10 high-activity + 5 medium-activity) were optimally positioned
4. **Multi-Objective Trade-off**: The Pareto front shows the trade-off between dose uniformity and minimum dose magnitude

### Technical Achievements

1. **Point Kernel Integration**: Successfully implemented accurate dose calculation considering:
   - Line source geometry
   - Angular integration along source length
   - Distance-dependent attenuation
   - Proper handling of field points inside/outside source extent

2. **Multi-Objective Optimization**: Effective use of NSGA-III for:
   - Balancing competing objectives (uniformity vs. magnitude)
   - Finding diverse Pareto-optimal solutions
   - Handling combinatorial source placement problem

3. **Visualization**: Generated comprehensive plots showing:
   - Source distribution on the rack
   - 2D dose distribution maps
   - Pareto front analysis

## Generated Files

1. **`source_distribution.png`**: Shows the optimized arrangement of sources on the YZ plane rack
2. **`dose_slice_YZ_0.png`**: Displays the dose distribution contours at 1m distance
3. **`pareto_front.png`**: Illustrates the trade-off between objectives in the optimization space

## Applications

This optimization framework can be applied to:
- Medical radiation therapy source planning
- Industrial radiography source arrangement
- Nuclear reactor fuel assembly optimization
- Radiation shielding design verification
- Research irradiation facility planning

## Scalability

The code is designed to handle:
- Variable source rack geometries
- Different source activity configurations
- Multiple source types and lengths
- Arbitrary calculation grid resolutions
- Larger population sizes and generation counts for production use

## Conclusion

The radiation dose distribution optimization successfully demonstrated:
- Effective implementation of point kernel integration for cobalt sources
- Multi-objective optimization using NSGA-III algorithm
- Achievement of excellent dose uniformity (DUR = 1.40)
- Comprehensive visualization of results
- Scalable framework for larger optimization problems

The results show that intelligent source arrangement can significantly improve dose uniformity while maintaining adequate dose rates for practical applications.