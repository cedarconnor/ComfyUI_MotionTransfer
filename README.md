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

### **Pipeline B2: CoTracker Mesh-Warp** (Advanced - New!)
- Track 4K-70K points using Meta's CoTracker (ECCV 2024)
- Build deformation mesh from point trajectories
- Transformer-based temporal stability (tracks entire video)
- Handles occlusions and complex organic motion

**Best for:** Temporal stability, organic motion, large deformations, character faces/hands

### **Pipeline C: 3D-Proxy** (Experimental)
- Monocular depth estimation
- 3D proxy reprojection with parallax handling

**Best for:** Camera motion, significant parallax, architectural shots

## Installation

**Super Simple - Just 2 Steps!**

1. Clone into ComfyUI custom_nodes:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/cedarconnor/ComfyUI_MotionTransfer.git
```

2. Install dependencies:
```bash
cd ComfyUI_MotionTransfer
pip install -r requirements.txt
```

3. Restart ComfyUI

**That's it!** RAFT code is now bundled directly in the package - no manual repository cloning required!

### What You Get Out of the Box

✅ **Optical Flow Model (RAFT - Bundled)**
- RAFT code included - works immediately
- Model weights (~100MB) - manual download required (see below)

✅ **All Pipeline A & B Nodes** - Ready to use immediately

### Download RAFT Model Weights

After installation, download the RAFT model weights (choose one):

**Option 1: Automatic Download (Recommended)**
```bash
# Windows PowerShell:
cd ComfyUI/models
mkdir raft
cd raft
Invoke-WebRequest -Uri "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip" -OutFile "models.zip"
Expand-Archive -Path "models.zip" -DestinationPath "." -Force

# Linux/Mac:
cd ComfyUI/models
mkdir -p raft
cd raft
wget https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
unzip models.zip
```

**Option 2: Manual Download**
1. Download from: https://github.com/princeton-vl/RAFT#demos
2. Save `raft-sintel.pth` to `ComfyUI/models/raft/`

### Optional: Pipeline B2 (CoTracker Mesh-Warp)

If you want to use Pipeline B2 (transformer-based point tracking for temporal stability):

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/s9roll7/comfyui_cotracker_node.git
```

CoTracker models (~500MB) auto-download from torch.hub on first use.

### Detailed Installation Guide

**Step-by-Step:**

1. **Navigate to ComfyUI custom_nodes directory:**
   ```bash
   cd ComfyUI/custom_nodes
   ```

2. **Clone this repository:**
   ```bash
   git clone https://github.com/cedarconnor/ComfyUI_MotionTransfer.git
   ```

   This will create a `ComfyUI_MotionTransfer` folder containing:
   - `motion_transfer_nodes.py` - Main node code
   - `raft_vendor/` - Bundled RAFT optical flow code
   - `searaft_vendor/` - Bundled SEA-RAFT optical flow code
   - `requirements.txt` - Python dependencies

3. **Install Python dependencies:**
   ```bash
   cd ComfyUI_MotionTransfer
   pip install -r requirements.txt
   ```

   This installs:
   - `huggingface-hub` - For downloading SEA-RAFT model weights
   - `imageio`, `scipy`, `tqdm` - For image/video processing
   - (Note: `torch`, `numpy`, `opencv`, `pillow` are already in ComfyUI)

4. **Restart ComfyUI:**
   - Close ComfyUI completely
   - Start it again
   - Check the console for: `[Motion Transfer] Using RAFT from: ...` (confirms vendored code loaded)

5. **First Run - Model Weights Download:**

   When you run your first workflow:
   - **SEA-RAFT models**: Auto-download from HuggingFace (~100-200MB)
   - **RAFT models**: You'll need to download manually (see below)

**RAFT Model Weights (if using raft-sintel/raft-things/raft-small):**

```bash
# Create models directory
mkdir -p ComfyUI/models/raft

# Download weights (choose one):
# raft-sintel (recommended for natural videos)
wget https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip -O models.zip
unzip models.zip -d ComfyUI/models/raft/

# Or download individually:
# raft-things.pth, raft-sintel.pth, raft-small.pth
```

### What Changed from Previous Versions?

**Old Installation (v0.1-v0.4):**
- ❌ Required manually cloning RAFT repository
- ❌ Required adding RAFT path to sys.path
- ❌ Complex setup with multiple steps

**New Installation (v0.5+):**
- ✅ RAFT code bundled automatically
- ✅ Single git clone
- ✅ Works out of the box
- ✅ No external repository dependencies

**Why This Change?**
- Simpler user experience (1 clone instead of 2)
- Ensures version compatibility
- Reduces installation errors
- RAFT code is lightweight (~50KB)

### External Dependencies Still Required

**Only CoTracker (Pipeline B2)** requires external installation:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/s9roll7/comfyui_cotracker_node.git
```

This is kept separate because:
- It's optional (Pipeline B2 only)
- It has its own dependencies and models
- Not all users need point tracking

## Node Reference

### Pipeline A Nodes (Flow-Warp)

#### RAFTFlowExtractor
- **Input:** Video frames (from stock ComfyUI video loader)
- **Output:** Flow fields, confidence/uncertainty maps
- **Parameters:**
  - `raft_iters`: Refinement iterations (6-8 for SEA-RAFT, 12-20 for RAFT)
  - `model_name`: Model variant (SEA-RAFT or RAFT)

**Model Selection Guide:**

| Model | Speed | Quality | VRAM | Best For |
|-------|-------|---------|------|----------|
| **sea-raft-small** | Fastest | Good | 8GB | Quick iterations, preview |
| **sea-raft-medium** ⭐ | Fast | Excellent | 12-24GB | **Recommended for most users** |
| **sea-raft-large** | Medium | Best | 24GB+ | Highest quality output |
| raft-sintel | Slow | Good | 12GB+ | Legacy workflows |
| raft-things | Slow | Fair | 12GB+ | Synthetic data |
| raft-small | Medium | Fair | 8GB+ | Faster RAFT variant |

**Performance Comparison (1080p→16K, 120 frames on RTX 4090):**
- SEA-RAFT Medium: ~6 minutes total (~3 sec/frame)
- RAFT Sintel: ~14 minutes total (~7 sec/frame)
- **Speedup: 2.3x faster with SEA-RAFT**

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

### Pipeline B2 Nodes (CoTracker Mesh-Warp)

Pipeline B2 uses the external CoTracker node plus one new node for mesh conversion. All downstream nodes (BarycentricWarp, TemporalConsistency, HiResWriter) are shared with Pipeline B.

#### CoTrackerNode (External)
- **Source:** [s9roll7/comfyui_cotracker_node](https://github.com/s9roll7/comfyui_cotracker_node)
- **Input:** Video frames, optional tracking points
- **Output:** JSON trajectory data, visualization
- **Parameters:**
  - `grid_size`: Grid density (20-64) - higher = more tracking points
  - `max_num_of_points`: Maximum points to track (100-4096)
  - `confidence_threshold`: Filter unreliable tracks (0.9)
  - `min_distance`: Minimum spacing between points (30)
  - `enable_backward`: Bidirectional tracking for occlusions

**Model:** Uses CoTracker3 (Meta AI, ECCV 2024) - auto-downloads from torch.hub

#### MeshFromCoTracker (New)
- **Input:** CoTracker JSON trajectory data
- **Output:** Deformation mesh sequence (compatible with BarycentricWarp)
- **Parameters:**
  - `frame_index`: Reference frame for UV coordinates (0)
  - `min_triangle_area`: Filter degenerate triangles (100.0)
  - `video_width/height`: Original video resolution

**Technical Details:**
- Converts sparse point tracks → triangulated mesh using Delaunay
- Same mesh format as MeshBuilder2D (vertices, faces, UVs)
- Filters small/degenerate triangles to prevent artifacts
- UV coordinates normalized to [0,1] for high-res warping

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

### Basic Flow-Warp Pipeline (with SEA-RAFT)

```
1. LoadVideo -> images
2. RAFTFlowExtractor(images, model="sea-raft-medium", iters=8) -> flow, confidence
3. LoadImage (16K still) -> still_image
4. FlowSRRefine(flow, still_image) -> flow_upscaled
5. FlowToSTMap(flow_upscaled) -> stmap
6. TileWarp16K(still_image, stmap) -> warped_sequence
7. TemporalConsistency(warped_sequence, flow_upscaled) -> stabilized
8. HiResWriter(stabilized) -> output files
```

### Available Example Workflows

See `examples/` directory for complete workflow JSON files:

- **`workflow_pipeline_a_searaft.json`** - Flow-Warp with SEA-RAFT (recommended)
- **`workflow_pipeline_a_flow.json`** - Flow-Warp with original RAFT
- **`workflow_pipeline_b_mesh.json`** - Mesh-Warp for large deformations
- **`workflow_pipeline_b2_cotracker.json`** - CoTracker Mesh-Warp for temporal stability (new!)
- **`workflow_pipeline_c_proxy.json`** - 3D-Proxy for parallax (experimental)
- **`README.md`** - Detailed workflow usage guide

## Pipeline Comparison

### When to Use Pipeline B vs B2

| Feature | **Pipeline B (RAFT Mesh)** | **Pipeline B2 (CoTracker Mesh)** |
|---------|---------------------------|----------------------------------|
| **Tracking Method** | Optical flow (frame-to-frame) | Sparse point tracking (whole video) |
| **Temporal Stability** | Good | **Excellent** (transformer sees full sequence) |
| **Occlusion Handling** | Limited | **Excellent** (tracks through occlusions) |
| **Setup Complexity** | Built-in | Requires external CoTracker node |
| **Processing Speed** | Fast (~1.4x real-time) | Medium (~1.0x real-time) |
| **VRAM Usage** | Moderate (12-24GB) | **Lower** (8-12GB for grid_size=64) |
| **Best For** | General mesh warping | Face/hand animation, organic motion |
| **Point Density** | Fixed grid | 100-4096 adaptive points |

**Recommendation:**
- **Start with Pipeline B** if you're new to mesh warping or want faster iterations
- **Use Pipeline B2** when temporal stability is critical (faces, hands, cloth)
- Both pipelines share the same BarycentricWarp node, so you can experiment

## Performance Tips

### Memory Management (16K images)
- 16K RGBA float32 = ~3GB per frame
- Use tile_size=2048, overlap=128 for 24GB VRAM
- Reduce tile_size to 1024 for 12GB VRAM
- Enable CPU offloading if needed

### Speed Optimization
- **Use SEA-RAFT instead of RAFT (2.3x faster)**
- Use fewer iterations: 6-8 for SEA-RAFT, 12 for RAFT
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

**RAFT/SEA-RAFT import error:**
- **This should never happen!** RAFT/SEA-RAFT code is bundled with the package
- If you see this error, please report it at: https://github.com/cedarconnor/ComfyUI_MotionTransfer/issues
- As a workaround, check that `raft_vendor/` and `searaft_vendor/` directories exist in the package

**SEA-RAFT model download fails:**
- Check internet connection (models download from HuggingFace on first use)
- Check CUDA compatibility with your PyTorch version (PyTorch >= 2.2.0 required)
- Install/upgrade huggingface-hub: `pip install -U huggingface-hub`
- Fallback to original RAFT models if needed (select "raft-sintel" instead of "sea-raft-medium")

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

### Optical Flow Models
- **SEA-RAFT**: [Simple, Efficient, Accurate RAFT for Optical Flow](https://github.com/princeton-vl/SEA-RAFT) (Wang, Lipson, Deng - ECCV 2024, Best Paper Award Candidate) - BSD-3-Clause License
- **RAFT**: [Recurrent All-Pairs Field Transforms for Optical Flow](https://github.com/princeton-vl/RAFT) (Teed & Deng, ECCV 2020) - BSD-3-Clause License

### Other Components
- Guided filtering: Fast Guided Filter (He et al., 2015)
- Mesh warping inspired by Lockdown/mocha
- Design document based on production VFX workflows

### Citations

If you use this in research, please cite:

**For SEA-RAFT:**
```bibtex
@inproceedings{wang2024searaft,
  title={SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow},
  author={Wang, Yihan and Lipson, Lahav and Deng, Jia},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

**For original RAFT:**
```bibtex
@inproceedings{teed2020raft,
  title={RAFT: Recurrent All-Pairs Field Transforms for Optical Flow},
  author={Teed, Zachary and Deng, Jia},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

## License

MIT License - see LICENSE file

Note: SEA-RAFT and RAFT are licensed under BSD-3-Clause and must be installed separately.
