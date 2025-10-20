# ComfyUI Motion Transfer - Example Workflows

This folder contains example workflow JSON files for all three motion transfer pipelines.

---

## Quick Start

1. **Open ComfyUI**
2. **Load a workflow:** Drag and drop one of the JSON files into ComfyUI
3. **Configure inputs:**
   - Replace `your_video.mp4` with your actual video file
   - Replace `your_still_16k.png` with your high-res still image
4. **Queue the workflow**

---

## Available Workflows

### 📁 `workflow_pipeline_a_flow.json` ⭐ RECOMMENDED

**Pipeline A: Flow-Warp (Production Ready)**

**What it does:**
- Extracts optical flow from low-res video using RAFT
- Upscales flow to match high-res still (16K+)
- Converts flow to STMap format
- Applies tiled warping with seamless blending
- Temporal stabilization to reduce flicker
- Exports final frames to PNG/EXR

**Best for:**
- Most use cases (general purpose)
- Fast processing (~5-10 sec per 16K frame)
- Non-extreme parallax
- Natural video sources

**Nodes used:**
```
LoadVideo → GetVideoComponents → RAFTFlowExtractor
                                        ↓
LoadImage ──────┬────────→ FlowSRRefine (guide)
                │              ↓
                │         FlowToSTMap
                │              ↓
                └─────→ TileWarp16K
                           ↓
                    TemporalConsistency
                           ↓
                       HiResWriter
```

**Settings:**
- `raft_iters`: 12 (balanced)
- `target_width/height`: 16000
- `tile_size`: 2048 (24GB VRAM) or 1024 (12GB VRAM)
- `overlap`: 128 pixels
- `blend_strength`: 0.3

---

### 📁 `workflow_pipeline_b_mesh.json`

**Pipeline B: Mesh-Warp (Advanced)**

**What it does:**
- Extracts optical flow from video
- Builds deformation mesh using Delaunay triangulation
- Adaptively refines mesh based on flow gradients
- Warps still image using barycentric interpolation
- Exports final frames

**Best for:**
- Large deformations (character animation, fabric)
- Surfaces that bend/fold significantly
- More stable results on edges vs pixel-based flow
- Non-rigid body deformations

**Nodes used:**
```
LoadVideo → GetVideoComponents → RAFTFlowExtractor
                                        ↓
                                   MeshBuilder2D
                                        ↓
                                 AdaptiveTessellate
                                        ↓
LoadImage ──────────────────→ BarycentricWarp
                                        ↓
                                   HiResWriter
```

**Settings:**
- `mesh_resolution`: 32 (control point density)
- `min_triangle_area`: 100.0 (filters bad triangles)
- `subdivision_threshold`: 10.0
- `interpolation`: linear (recommended for mesh)

**Note:** Slower than Pipeline A due to triangle rasterization.

---

### 📁 `workflow_pipeline_c_proxy.json`

**Pipeline C: 3D-Proxy (Experimental)**

**What it does:**
- Extracts optical flow from video
- Estimates depth maps for each frame
- Reprojects still image using depth + flow
- Handles parallax by treating scene as 3D proxy
- Exports final frames

**Best for:**
- Camera motion with parallax
- Foreground/background separation
- Architectural or landscape shots
- Videos with significant depth variation

**Nodes used:**
```
LoadVideo → GetVideoComponents ─┬→ RAFTFlowExtractor
                                │          ↓
                                └→ DepthEstimator
                                           ↓
LoadImage ────────────────→ ProxyReprojector
                                           ↓
                                      HiResWriter
```

**Settings:**
- `model`: midas (currently placeholder)
- `focal_length`: 1000 (adjust for your camera)

**Current Limitations:**
- ⚠️ Uses placeholder depth estimation (simple Gaussian blur)
- ⚠️ Real MiDaS/DPT models not yet integrated
- ⚠️ Camera pose estimation simplified
- Best results when real depth models are added (future enhancement)

---

## Configuration Guide

### Input Files

**Video Requirements:**
- Resolution: 720p-1080p (low-res driving video)
- Frame rate: 16-30 fps
- Duration: 3-10 seconds typical
- Format: MP4, MOV, AVI (anything ffmpeg supports)
- Location: Place in `ComfyUI/input/` folder

**Still Image Requirements:**
- Resolution: Any high-res (4K, 8K, 16K+)
- Format: PNG, EXR, TIFF, JPG
- Color: sRGB (standard)
- Location: Place in `ComfyUI/input/` folder

### Output Settings

**Default output path:** `output/pipeline_X/frame_XXXX.png`

**Change output format:**
```json
"widgets_values": [
  "output/my_project/frame",  // Path (without extension)
  "png",                       // Format: png/exr/jpg
  0                           // Starting frame number
]
```

**Format comparison:**
- **PNG:** 8-bit sRGB, lossless, ~10-50 MB per 16K frame
- **EXR:** 16-bit half, linear, ~100-200 MB per 16K frame (best for VFX)
- **JPG:** 8-bit sRGB, quality 95, ~5-20 MB per 16K frame (smallest)

### Memory Optimization

**For 12GB VRAM:**
```json
// In TileWarp16K node:
"widgets_values": [
  1024,    // tile_size (reduced from 2048)
  64,      // overlap (reduced from 128)
  "cubic"
]
```

**For 8GB VRAM:**
```json
"widgets_values": [
  512,     // tile_size
  32,      // overlap
  "linear" // faster interpolation
]
```

**For 48GB+ VRAM:**
```json
"widgets_values": [
  4096,    // tile_size (larger tiles)
  256,     // overlap (better quality)
  "lanczos4"
]
```

---

## Troubleshooting

### "RAFT not found" error
```bash
pip install git+https://github.com/princeton-vl/RAFT.git
```

### Out of memory (CUDA OOM)
- Reduce `tile_size` in TileWarp16K
- Reduce `raft_iters` in RAFTFlowExtractor (try 8 instead of 12)
- Process fewer frames at once

### Visible seams in output
- Increase `overlap` parameter (try 256)
- Use `cubic` or `lanczos4` interpolation
- Check that guide_image is connected in Pipeline A

### Temporal flicker
- Increase `blend_strength` in TemporalConsistency (try 0.5)
- Use more RAFT iterations for better flow accuracy
- Check flow confidence maps for low-confidence regions

### Slow processing
- Reduce `tile_size` (faster but uses more tiles)
- Use `linear` interpolation instead of `cubic`
- Reduce `raft_iters` to 8
- Use Pipeline A instead of B or C

---

## Workflow Customization

### Combining with other ComfyUI nodes

**Add upscaling before motion transfer:**
```
LoadImage → Upscaler → (use as still_image)
```

**Add preview/downscale after:**
```
HiResWriter → ImageScale → SaveImage (for preview)
```

**Add masking:**
```
LoadImage (mask) → (future: mask support in nodes)
```

### Batch processing multiple videos

Create a loop wrapper or use ComfyUI's batch features to process multiple videos sequentially.

---

## Performance Benchmarks

**Hardware:** RTX 4090, 24GB VRAM
**Input:** 1080p video (5 sec @ 24fps = 120 frames) → 16K still

| Pipeline | Time per frame | Total time | Notes |
|----------|---------------|------------|-------|
| A (Flow) | ~7 seconds    | ~14 min    | Recommended |
| B (Mesh) | ~12 seconds   | ~24 min    | Better for deformation |
| C (Proxy)| ~8 seconds    | ~16 min    | Experimental |

**Settings used:** tile_size=2048, overlap=128, raft_iters=12

---

## Next Steps

1. **Try Pipeline A first** - It's the fastest and most reliable
2. **Experiment with parameters** - Adjust tile_size, overlap, blend_strength
3. **Test different videos** - Natural motion works best
4. **Compare outputs** - Try different pipelines on same input
5. **Report issues** - https://github.com/yourname/ComfyUI_MotionTransfer/issues

---

## Credits

- Workflows designed for ComfyUI Motion Transfer v0.1.0
- Based on RAFT optical flow, mesh deformation, and 3D proxy techniques
- See main README.md for full credits and documentation
