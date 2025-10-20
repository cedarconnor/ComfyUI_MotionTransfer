# ComfyUI Motion Transfer Pack

Transfer motion from low-resolution AI-generated videos to ultra-high-resolution still images (up to 16K+).

## Features

This node pack provides three complementary pipelines for motion transfer:

### **Pipeline A: Flow-Warp** (Core - Recommended)
- Extract optical flow from low-res video using RAFT
- Upscale flow fields to match high-res still (with guided filtering)
- Convert to STMap format
- Apply tiled warping with seamless blending
- Temporal stabilization for flicker reduction

**Best for:** General purpose, most video types, fast processing

### **Pipeline B: Mesh-Warp** (Advanced)
- Build 2D deformation mesh from optical flow
- Adaptive tessellation based on flow gradients
- Barycentric warping for stable deformation

**Best for:** Large deformations, character animation, fabric/cloth

### **Pipeline C: 3D-Proxy** (Experimental)
- Monocular depth estimation
- 3D proxy reprojection with parallax handling

**Best for:** Camera motion, significant parallax, architectural shots

## Installation

1. Clone into ComfyUI custom_nodes:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourname/ComfyUI_MotionTransfer.git
cd ComfyUI_MotionTransfer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install RAFT (required for optical flow):
```bash
pip install git+https://github.com/princeton-vl/RAFT.git
```

4. Download RAFT pretrained models:
```bash
# Place models in ComfyUI/models/raft/
wget https://github.com/princeton-vl/RAFT/releases/download/v1.0/raft-sintel.pth
```

5. Restart ComfyUI

## Node Reference

### Pipeline A Nodes (Flow-Warp)

#### RAFTFlowExtractor
- **Input:** Video frames (from stock ComfyUI video loader)
- **Output:** Flow fields, confidence maps
- **Parameters:**
  - `raft_iters`: Refinement iterations (12-20)
  - `model_name`: RAFT model variant

#### FlowSRRefine
- **Input:** Low-res flow, high-res guide image
- **Output:** Upscaled and refined flow
- **Parameters:**
  - `target_width/height`: Output resolution (e.g., 16000)
  - `guided_filter_radius`: Edge-aware smoothing (8-16)

#### FlowToSTMap
- **Input:** Flow fields
- **Output:** Normalized STMap (Nuke/AE compatible)

#### TileWarp16K
- **Input:** High-res still, STMap sequence
- **Output:** Warped frame sequence
- **Parameters:**
  - `tile_size`: Processing tile size (2048)
  - `overlap`: Blend overlap (128)
  - `interpolation`: cubic/linear/lanczos4

#### TemporalConsistency
- **Input:** Warped frames, flow fields
- **Output:** Temporally stabilized sequence
- **Parameters:**
  - `blend_strength`: Temporal blending (0.3)

#### HiResWriter
- **Input:** Image sequence
- **Output:** Saves to disk
- **Parameters:**
  - `format`: png/exr/jpg
  - `output_path`: File path pattern

### Pipeline B Nodes (Mesh-Warp)

#### MeshBuilder2D
- **Input:** Flow fields
- **Output:** Deformation mesh sequence
- **Parameters:**
  - `mesh_resolution`: Mesh density (32)
  - `min_triangle_area`: Triangle filtering (100.0)

#### AdaptiveTessellate
- **Input:** Mesh, flow gradients
- **Output:** Refined mesh
- **Parameters:**
  - `subdivision_threshold`: Refinement sensitivity (10.0)
  - `max_subdivisions`: Max iterations (2)

#### BarycentricWarp
- **Input:** High-res still, mesh sequence
- **Output:** Warped sequence
- **Parameters:**
  - `interpolation`: linear/cubic

### Pipeline C Nodes (3D-Proxy)

#### DepthEstimator
- **Input:** Video frames
- **Output:** Depth maps
- **Parameters:**
  - `model`: midas/dpt

#### ProxyReprojector
- **Input:** High-res still, depth maps, flow
- **Output:** Reprojected sequence
- **Parameters:**
  - `focal_length`: Camera focal length (1000.0)

## Example Workflows

### Basic Flow-Warp Pipeline

```
1. LoadVideo -> images
2. RAFTFlowExtractor(images) -> flow, confidence
3. LoadImage (16K still) -> still_image
4. FlowSRRefine(flow, still_image) -> flow_upscaled
5. FlowToSTMap(flow_upscaled) -> stmap
6. TileWarp16K(still_image, stmap) -> warped_sequence
7. TemporalConsistency(warped_sequence, flow_upscaled) -> stabilized
8. HiResWriter(stabilized) -> output files
```

See `examples/` directory for complete workflow JSON files.

## Performance Tips

### Memory Management (16K images)
- 16K RGBA float32 = ~3GB per frame
- Use tile_size=2048, overlap=128 for 24GB VRAM
- Reduce tile_size to 1024 for 12GB VRAM
- Enable CPU offloading if needed

### Speed Optimization
- Use RAFT with fewer iterations (8-12) for faster flow extraction
- Use "linear" interpolation instead of "cubic" for warping
- Process shorter sequences (3-5 seconds)
- Multi-GPU: Split time-range across GPUs

### Quality Settings
- For best quality:
  - `raft_iters`: 20
  - `guided_filter_radius`: 16
  - `tile_size`: 4096 (if VRAM allows)
  - `overlap`: 256
  - `interpolation`: lanczos4

## Troubleshooting

**Seams visible in output:**
- Increase `overlap` parameter (128-256)
- Check STMap continuity across tiles
- Use guided filter to smooth flow

**Temporal flicker:**
- Increase `blend_strength` in TemporalConsistency
- Use forward-backward flow consistency
- Check flow confidence values

**Out of memory:**
- Reduce `tile_size`
- Process fewer frames at once
- Use PNG instead of keeping frames in memory

**RAFT import error:**
- Ensure RAFT is installed: `pip install git+https://github.com/princeton-vl/RAFT.git`
- Check CUDA compatibility with your PyTorch version

## Technical Details

### Data Flow
- ComfyUI uses `[B, H, W, C]` tensor format (batch, height, width, channels)
- Flow fields: `[B-1, H, W, 2]` where channel 0=u (horizontal), 1=v (vertical)
- STMaps: `[B, H, W, 3]` where R=S, G=T, B=unused (normalized [0,1])

### Custom Types
- `FLOW`: Optical flow displacement fields
- `MESH`: Dictionary containing vertices, faces, UVs

## Roadmap

- [x] v0.1: Core flow/STMap pipeline (Pipeline A)
- [x] v0.2: Tiled warping with feathering
- [x] v0.3: Temporal consistency
- [x] v0.4: Mesh-based warping (Pipeline B)
- [x] v0.5: 3D proxy (Pipeline C - experimental)
- [ ] v0.6: CUDA kernels for critical paths
- [ ] v1.0: Production release with full docs

## Credits

- Optical flow: [RAFT](https://github.com/princeton-vl/RAFT) (Teed & Deng, ECCV 2020)
- Guided filtering: Fast Guided Filter (He et al., 2015)
- Mesh warping inspired by Lockdown/mocha
- Design document based on production VFX workflows

## License

MIT License - see LICENSE file
