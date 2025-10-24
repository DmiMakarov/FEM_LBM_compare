# FEM Mesh Performance Comparison

## Available Mesh Sizes

| Mesh Type | Dimensions | Nodes | Elements | Time/Step | Speedup | Use Case |
|-----------|------------|-------|----------|-----------|---------|----------|
| **Very Coarse** | 20×5 | 100 | 152 | ~0.04s | **650x** | Rapid testing, debugging |
| **Coarse** | 40×10 | 400 | 702 | ~0.65s | **38x** | Development, quick results |
| **Fine** | 100×25 | 2,500 | 4,752 | ~25s | 1x | Production, accurate results |

## Performance Summary

- **Very Coarse (20×5)**: Perfect for rapid testing and debugging
  - 20 time steps in ~0.8 seconds
  - Ideal for algorithm development

- **Coarse (40×10)**: Good balance of speed and accuracy
  - 20 time steps in ~13 seconds
  - Recommended for most development work

- **Fine (100×25)**: High accuracy but slow
  - 20 time steps in ~500 seconds (8+ minutes)
  - Use only for final production runs

## Usage

### Quick Testing
```bash
cd FEM
python run_fem_fast.py 20 very_coarse 20
```

### Development Work
```bash
cd FEM
python run_fem_fast.py 100 coarse 50
```

### Production Runs
```bash
cd FEM
python run_fem_fast.py 100 fine 100
```

## Mesh Quality vs Speed Trade-off

- **Very Coarse**: May miss fine flow features, but excellent for testing
- **Coarse**: Good representation of main flow features, much faster
- **Fine**: Captures all flow details, but computationally expensive

## Recommendations

1. **Start with very_coarse** for initial testing and debugging
2. **Use coarse** for most development and comparison work
3. **Use fine** only for final production results when accuracy is critical

The coarse mesh (40×10) provides the best balance of speed and accuracy for most applications.
