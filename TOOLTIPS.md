# Parameter Tooltips Reference

Complete reference of all parameter tooltips in ComfyUI Motion Transfer nodes.

---

## Pipeline A: Flow-Warp Nodes

### RAFTFlowExtractor

**images**
- Video frames from ComfyUI video loader. Expects [B, H, W, C] batch of images.

**raft_iters**
- Number of refinement iterations for RAFT. Higher values (16-20) give more accurate flow but take longer. Lower values (6-8) are faster but less precise. Default 12 is a good balance.

**model_name**
- RAFT model variant. 'raft-sintel': best for natural videos (recommended). 'raft-things': trained on synthetic data. 'raft-small': faster but less accurate.

---

### FlowSRRefine

**flow**
- Optical flow fields from RAFTFlowExtractor. Low-resolution flow to be upscaled.

**guide_image**
- High-resolution still image used as guidance for edge-aware filtering. Prevents flow from bleeding across sharp edges.

**target_width**
- Target width for upscaled flow (should match your high-res still width). Common: 4K=3840, 8K=7680, 16K=15360.

**target_height**
- Target height for upscaled flow (should match your high-res still height). Common: 4K=2160, 8K=4320, 16K=8640.

**guided_filter_radius**
- Radius for guided filter smoothing. Larger values (16-32) give smoother flow, smaller values (4-8) preserve detail better. 8 is a good default.

**guided_filter_eps**
- Regularization parameter for guided filter. Lower values (1e-4) preserve edges better, higher values (1e-2) give smoother results. 1e-3 is recommended.

---

### FlowToSTMap

**flow**
- High-resolution flow fields from FlowSRRefine. Will be converted to normalized STMap coordinates for warping.

---

### TileWarp16K

**still_image**
- High-resolution still image to warp. This is the 16K (or other high-res) image that will have motion applied to it.

**stmap**
- STMap sequence from FlowToSTMap. Contains normalized UV coordinates that define how to warp each pixel.

**tile_size**
- Size of processing tiles. Larger tiles (4096) are faster but need more VRAM. Use 2048 for 24GB GPU, 1024 for 12GB GPU, 512 for 8GB GPU.

**overlap**
- Overlap between tiles for blending. Larger values (256) give smoother seams but slower processing. 128 is recommended, use 64 minimum.

**interpolation**
- Interpolation method. 'cubic': best quality/speed balance (recommended). 'linear': fastest but lower quality. 'lanczos4': highest quality but slowest.

---

### TemporalConsistency

**frames**
- Warped frame sequence from TileWarp16K. These frames will be temporally stabilized to reduce flicker.

**flow**
- High-resolution flow fields from FlowSRRefine. Used to warp previous frame forward for temporal blending.

**blend_strength**
- Temporal blending strength. 0.0 = no blending (may flicker), 0.3 = balanced (recommended), 0.5+ = strong smoothing (may blur motion). Reduce if motion looks ghosted.

---

### HiResWriter

**images**
- Final image sequence to export (typically from TemporalConsistency). Will be written to disk as individual frames.

**output_path**
- Output file path pattern (without extension). Example: 'C:/renders/shot01/frame' will create frame_0000.png, frame_0001.png, etc. Directory will be created if needed.

**format**
- Output format. 'png': 8-bit sRGB, lossless (recommended for web/preview). 'exr': 16-bit half float, linear (best for VFX/compositing). 'jpg': 8-bit sRGB, quality 95 (smallest files).

**start_frame**
- Starting frame number for file naming. Use 0 for frame_0000, 1001 for film standard (frame_1001), etc.

---

## Pipeline B: Mesh-Warp Nodes

### MeshBuilder2D

**flow**
- Optical flow fields from RAFTFlowExtractor. Flow will be sampled at mesh vertices to create deformation mesh.

**mesh_resolution**
- Number of mesh control points along each axis. Higher values (64-128) give finer deformation control but slower. Lower values (16-32) are faster. 32 is a good balance.

**min_triangle_area**
- Minimum area for triangles (in pixels²). Filters out degenerate/tiny triangles that can cause artifacts. Lower values keep more triangles but may have issues. 100.0 is recommended.

---

### AdaptiveTessellate

**mesh_sequence**
- Mesh sequence from MeshBuilder2D to be refined with adaptive subdivision.

**flow**
- Flow fields used to compute gradient magnitude for adaptive subdivision. Areas with high flow gradients get more subdivision.

**subdivision_threshold**
- Flow gradient threshold for triggering subdivision. Lower values (5.0) subdivide more aggressively, higher values (20.0) subdivide less. Currently placeholder - full subdivision not yet implemented.

**max_subdivisions**
- Maximum subdivision iterations. Higher values create finer meshes but slower processing. 0 = no subdivision, 2 = balanced, 4 = very detailed. Currently placeholder.

---

### BarycentricWarp

**still_image**
- High-resolution still image to warp using mesh deformation. Alternative to TileWarp16K - better for large deformations.

**mesh_sequence**
- Mesh sequence from AdaptiveTessellate (or MeshBuilder2D). Contains deformed triangles that define the warping.

**interpolation**
- Interpolation method for triangle warping. 'linear': faster (recommended for mesh). 'cubic': higher quality but slower and may cause artifacts with meshes.

---

## Pipeline C: 3D-Proxy Nodes

### DepthEstimator

**images**
- Video frames from ComfyUI video loader. Depth will be estimated for each frame to enable parallax-aware warping.

**model**
- Depth estimation model. 'midas': MiDaS (lighter, faster). 'dpt': DPT (more accurate). Currently placeholder - real models not yet integrated, uses simple Gaussian blur.

---

### ProxyReprojector

**still_image**
- High-resolution still image to reproject using depth information. Depth enables parallax-correct warping for camera motion.

**depth_maps**
- Depth map sequence from DepthEstimator. Closer objects (brighter) move more than distant objects (darker) under camera motion.

**flow**
- Optical flow from RAFTFlowExtractor. Combined with depth to estimate camera motion and create parallax-aware warping.

**focal_length**
- Estimated camera focal length in pixels. Higher values (2000-5000) = telephoto (less perspective). Lower values (500-1000) = wide angle (more perspective). Affects parallax strength.

---

## Quick Reference Tables

### Memory Requirements (tile_size)

| VRAM | Recommended tile_size | Max resolution |
|------|----------------------|----------------|
| 8GB  | 512                  | 8K             |
| 12GB | 1024                 | 12K            |
| 24GB | 2048                 | 16K+           |
| 48GB | 4096                 | 32K+           |

### Quality Settings (raft_iters)

| Setting | raft_iters | Speed | Quality |
|---------|-----------|-------|---------|
| Draft   | 6-8       | Fast  | Low     |
| Normal  | 12        | Medium| Good    |
| High    | 16-20     | Slow  | Best    |

### Overlap Guidelines

| Image Size | Minimum overlap | Recommended | Best quality |
|-----------|----------------|-------------|--------------|
| 4K        | 32             | 64          | 128          |
| 8K        | 64             | 128         | 256          |
| 16K+      | 64             | 128         | 256          |

### Temporal Blending (blend_strength)

| Value | Effect                          | Use case                    |
|-------|--------------------------------|-----------------------------|
| 0.0   | No blending                    | Sharp motion, may flicker   |
| 0.2   | Light smoothing                | Fast motion                 |
| 0.3   | Balanced (recommended)         | Most cases                  |
| 0.5   | Strong smoothing               | Very jerky motion           |
| 0.7+  | Very strong (may blur motion)  | Extreme flicker reduction   |

---

## Common Parameter Combinations

### For Web/Preview (Fast)
- raft_iters: 8
- tile_size: 1024
- overlap: 64
- interpolation: linear
- blend_strength: 0.3
- format: jpg

### For Production (Balanced)
- raft_iters: 12
- tile_size: 2048
- overlap: 128
- interpolation: cubic
- blend_strength: 0.3
- format: png

### For VFX/Compositing (Best)
- raft_iters: 20
- tile_size: 2048-4096
- overlap: 256
- interpolation: lanczos4
- blend_strength: 0.2
- format: exr

### For Low VRAM (12GB)
- raft_iters: 12
- tile_size: 1024
- overlap: 64
- interpolation: cubic
- blend_strength: 0.3
- format: png
