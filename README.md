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

**Super Simple - Just 3 Steps!**

1. **Clone into ComfyUI custom_nodes:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/cedarconnor/ComfyUI_MotionTransfer.git
```

2. **Install dependencies:**
```bash
cd ComfyUI_MotionTransfer
pip install -r requirements.txt
```

3. **Restart ComfyUI**

**That's it!** Both RAFT and SEA-RAFT code are bundled directly in the package - no external repository cloning required!

### What You Get Out of the Box

✅ **Optical Flow Models (Both Bundled!)**
- **RAFT** code included in `raft_vendor/` - works immediately
- **SEA-RAFT** code included in `searaft_vendor/` - works immediately
- No external repository cloning needed!

✅ **All Pipeline A & B Nodes** - Ready to use immediately

✅ **Clean Modular Architecture (v0.6.0+)**
- Simplified model loading (no sys.path manipulation)
- Automatic model type detection
- Clear error messages

### Model Weights Download

You need to download model weights to use the optical flow models:

#### Option 1: SEA-RAFT (Recommended - Auto-Download)

**No manual download needed!** SEA-RAFT models auto-download from HuggingFace on first use:
- `sea-raft-small`: ~100MB (8GB VRAM)
- `sea-raft-medium`: ~150MB (12-24GB VRAM) ⭐ **Recommended**
- `sea-raft-large`: ~200MB (24GB+ VRAM)

**Requirements:**
- `huggingface-hub` (installed via requirements.txt)
- Internet connection for first-time download
- Models cache to `~/.cache/huggingface/` for future use

**Advantages:**
- ✅ 2.3x faster than RAFT
- ✅ 22% more accurate (ECCV 2024 Best Paper Candidate)
- ✅ Auto-downloads (no manual steps)
- ✅ Better edge preservation

#### Option 2: RAFT (Manual Download Required)

If you prefer original RAFT or need offline installation:

**Windows PowerShell:**
```powershell
cd ComfyUI/models
mkdir raft
cd raft
Invoke-WebRequest -Uri "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip" -OutFile "models.zip"
Expand-Archive -Path "models.zip" -DestinationPath "." -Force
```

**Linux/Mac:**
```bash
cd ComfyUI/models
mkdir -p raft
cd raft
wget https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
unzip models.zip
```

**Manual Download:**
1. Download from: https://github.com/princeton-vl/RAFT#demos
2. Save `raft-sintel.pth`, `raft-things.pth`, or `raft-small.pth` to `ComfyUI/models/raft/`

**When to use RAFT:**
- ⚠️ Legacy workflows (backward compatibility)
- ⚠️ Offline installation (no internet access)
- ⚠️ Older PyTorch versions (< 2.2.0)

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

   This creates `ComfyUI_MotionTransfer/` containing:
   - `motion_transfer_nodes.py` - Main node implementations
   - `models/` - Modular model loaders (new in v0.6.0)
   - `raft_vendor/` - Bundled RAFT code (no external repo needed!)
   - `searaft_vendor/` - Bundled SEA-RAFT code (no external repo needed!)
   - `requirements.txt` - Python dependencies

3. **Install Python dependencies:**
   ```bash
   cd ComfyUI_MotionTransfer
   pip install -r requirements.txt
   ```

   This installs:
   - `huggingface-hub` - For SEA-RAFT auto-downloads
   - `imageio`, `scipy`, `tqdm` - Image/video processing utilities
   - (Note: `torch`, `numpy`, `opencv`, `pillow` already in ComfyUI)

4. **Restart ComfyUI:**
   - Close ComfyUI completely
   - Start it again
   - Check console for: `[RAFT Loader]` or `[SEA-RAFT Loader]` messages

5. **First Run - Choose Your Model:**

   **Recommended: SEA-RAFT** (auto-downloads on first use)
   - Select `sea-raft-medium` in RAFTFlowExtractor node
   - First run downloads ~150MB from HuggingFace
   - Subsequent runs use cached model
   - No manual setup required!

   **Alternative: RAFT** (manual download required)
   - Download weights (see "Option 2: RAFT" above)
   - Select `raft-sintel` in RAFTFlowExtractor node

### CUDA Acceleration (Optional - 5-10× Faster!)

**NEW!** GPU-accelerated kernels for critical nodes. Compilation optional - nodes work without CUDA.

**Performance Gains:**
| Node              | CPU Time (120 frames) | CUDA Time | Speedup |
|-------------------|-----------------------|-----------|---------|
| TileWarp16K       | ~20 min               | ~2-3 min  | **8-15×** |
| BarycentricWarp   | ~24 min               | ~2 min    | **10-20×** |
| FlowSRRefine      | ~2 min                | ~30 sec   | **3-5×**  |

**Total Pipeline Speedup:** 40 min → 5-7 min for typical 16K workflows!

**Quick Setup:**
1. Install [CUDA Toolkit 11.x or 12.x](https://developer.nvidia.com/cuda-downloads)
2. Compile kernels:
   ```bash
   cd ComfyUI/custom_nodes/ComfyUI_MotionTransfer/cuda
   build.bat  # Windows
   # OR
   ./build.sh  # Linux/macOS
   ```
3. Restart ComfyUI → see `[Motion Transfer] CUDA acceleration enabled`

**Full documentation:** [cuda/README.md](cuda/README.md) (installation, troubleshooting, benchmarks)

**Requirements:**
- NVIDIA GPU (GTX 1060+ / RTX 20xx+)
- 12-24GB VRAM (for 16K images)
- CUDA Toolkit + nvcc compiler

**Without CUDA:** Nodes automatically fall back to CPU (no errors, just slower)

### What's New in v0.6.0?

**Architecture Refactor:**
- ✅ **Modular model loaders** - Clean `models/` package structure
- ✅ **No sys.path manipulation** - Uses Python's proper import system
- ✅ **Simplified code** - 94% reduction in model loading complexity
- ✅ **Better error messages** - Clear, actionable guidance
- ✅ **Full dual-model support** - Both RAFT and SEA-RAFT working
- ✅ **CUDA acceleration** - Optional GPU kernels for 5-10× speedup

**Before (v0.5 and earlier):**
- 210 lines of complex path detection
- sys.path manipulation (fragile)
- Bundled code existed but wasn't used
- SEA-RAFT listed but broken

**After (v0.6.0):**
- 12 lines using clean model loaders
- Relative imports (proper Python)
- Bundled code actually used!
- Both RAFT and SEA-RAFT fully functional

**Inspiration:** Architecture inspired by [alanhzh/ComfyUI-RAFT](https://github.com/alanhzh/ComfyUI-RAFT) with enhanced dual-model support and HuggingFace integration.

### External Dependencies

**Only CoTracker (Pipeline B2)** requires external installation:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/s9roll7/comfyui_cotracker_node.git
```

**Why it's separate:**
- Optional (Pipeline B2 only)
- Has its own dependencies and models
- Not all users need point tracking

## Node Reference

### Pipeline A Nodes (Flow-Warp)

#### RAFTFlowExtractor
Extract dense optical flow between consecutive video frames using RAFT or SEA-RAFT.

**Inputs:**
- `images`: Video frames from ComfyUI video loader `[B, H, W, C]`

**Outputs:**
- `flow`: Optical flow fields `[B-1, H, W, 2]` (u, v displacement vectors)
- `confidence`: Flow confidence/uncertainty maps `[B-1, H, W, 1]`

**Parameters:**
- `raft_iters`: Refinement iterations
  - SEA-RAFT: 6-8 (auto-adjusts to 8 if left at default 12)
  - RAFT: 12-20
  - Higher = better quality but slower
- `model_name`: Model selection (see table below)

**Model Selection Guide:**

| Model | Speed | Quality | VRAM | Download | Best For |
|-------|-------|---------|------|----------|----------|
| **sea-raft-small** | Fastest | Good | 8GB | Auto (100MB) | Quick iterations, preview |
| **sea-raft-medium** ⭐ | Fast | Excellent | 12-24GB | Auto (150MB) | **Recommended - best balance** |
| **sea-raft-large** | Medium | Best | 24GB+ | Auto (200MB) | Highest quality output |
| raft-sintel | Slow | Good | 12GB+ | Manual | Legacy workflows, offline |
| raft-things | Slow | Fair | 12GB+ | Manual | Synthetic data training |
| raft-small | Medium | Fair | 8GB+ | Manual | Faster RAFT variant |

**Key Differences:**
- **SEA-RAFT (2024)**: Auto-download, 2.3x faster, 22% more accurate, better edges
- **RAFT (2020)**: Manual download, backward compatible, offline-friendly

**Performance Comparison (1080p→16K, 120 frames on RTX 4090):**
- SEA-RAFT Medium: ~6 minutes total (~3 sec/frame) ⚡
- RAFT Sintel: ~14 minutes total (~7 sec/frame)
- **Speedup: 2.3x faster with SEA-RAFT**

**Technical Notes:**
- Both use bundled vendor code (no external repos needed)
- SEA-RAFT provides native uncertainty output (better confidence maps)
- RAFT uses heuristic confidence based on flow magnitude
- Model auto-caching for performance

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

### Installation Issues

**"Cannot find module 'models'" error:**
- Ensure you cloned the complete repository (includes `models/` directory)
- Check that `models/__init__.py`, `models/raft_loader.py`, `models/searaft_loader.py` exist
- Restart ComfyUI after installation
- This should not happen with v0.6.0+ - report if it does!

**"RAFT checkpoint not found" error:**
- For RAFT models: Download weights to `ComfyUI/models/raft/` (see installation section)
- Check the path in error message - must match `ComfyUI/models/raft/raft-sintel.pth`
- Try absolute path: `C:\ComfyUI\models\raft\raft-sintel.pth` (Windows) or `/path/to/ComfyUI/models/raft/raft-sintel.pth` (Linux/Mac)
- For SEA-RAFT: Use `sea-raft-medium` instead (auto-downloads)

**"SEA-RAFT model download fails":**
- Check internet connection (models download from HuggingFace on first use)
- Verify `huggingface-hub` installed: `pip list | grep huggingface-hub`
- Check PyTorch version: `python -c "import torch; print(torch.__version__)"` (need >= 2.2.0 for SEA-RAFT)
- Install/upgrade: `pip install --upgrade huggingface-hub torch`
- Fallback: Use RAFT models instead (`raft-sintel`)

**"Import error" or "Module not found":**
- **This should never happen!** RAFT/SEA-RAFT code is bundled (v0.6.0+)
- Check that `raft_vendor/` and `searaft_vendor/` directories exist
- Verify `models/` directory with 3 files: `__init__.py`, `raft_loader.py`, `searaft_loader.py`
- Report at: https://github.com/cedarconnor/ComfyUI_MotionTransfer/issues

### Runtime Issues

**Seams visible in output:**
- Increase `overlap` parameter in TileWarp16K (128→256)
- Check STMap continuity across tiles
- Use guided filter to smooth flow (increase `guided_filter_radius`)

**Temporal flicker:**
- Increase `blend_strength` in TemporalConsistency (0.3→0.5)
- Try different models (SEA-RAFT has better temporal stability)
- Check flow confidence values (low confidence = potential flicker)

**Out of memory:**
- Reduce `tile_size` in TileWarp16K (2048→1024 or 512)
- Process fewer frames at once
- Use PNG output instead of keeping frames in memory
- Try SEA-RAFT models (lower VRAM usage than RAFT)

**Slow performance:**
- Use **SEA-RAFT instead of RAFT** (2.3x faster!)
- Reduce `raft_iters` (12→8 for SEA-RAFT, 20→12 for RAFT)
- Use `linear` interpolation instead of `cubic` or `lanczos4`
- Reduce `tile_size` for faster warping
- Process shorter sequences (3-5 seconds instead of 10+)

### Console Messages

**"[RAFT Loader] Using cached model":**
- ✅ Normal - model already loaded from previous run

**"[SEA-RAFT Loader] First run downloads model":**
- ✅ Normal - downloading from HuggingFace (happens once)

**"[Motion Transfer] Auto-adjusted iterations to 8 for SEA-RAFT":**
- ✅ Normal - optimization for SEA-RAFT's faster convergence

**"WARNING: opencv-contrib-python not found, using bilateral filter":**
- ⚠️ Non-critical - guided filter unavailable, using fallback
- Install for better quality: `pip install opencv-contrib-python`

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
- [x] v0.5: 3D proxy (Pipeline C - experimental) + CoTracker integration (Pipeline B2)
- [x] v0.6: **RAFT/SEA-RAFT architecture refactor** ⭐
  - Modular model loaders
  - Full dual-model support
  - Simplified codebase (94% reduction in model loading code)
  - SEA-RAFT HuggingFace integration
- [ ] v0.7: CUDA kernels for critical paths
- [ ] v0.8: GUI progress indicators and better UX
- [ ] v1.0: Production release with full docs

## Credits

### Optical Flow Models
- **SEA-RAFT**: [Simple, Efficient, Accurate RAFT for Optical Flow](https://github.com/princeton-vl/SEA-RAFT) (Wang, Lipson, Deng - ECCV 2024, Best Paper Award Candidate) - BSD-3-Clause License
- **RAFT**: [Recurrent All-Pairs Field Transforms for Optical Flow](https://github.com/princeton-vl/RAFT) (Teed & Deng, ECCV 2020) - BSD-3-Clause License

### Other Components
- **Architecture inspiration**: [alanhzh/ComfyUI-RAFT](https://github.com/alanhzh/ComfyUI-RAFT) for clean relative import approach
- Guided filtering: Fast Guided Filter (He et al., 2015)
- Mesh warping inspired by Lockdown/mocha
- Design document based on production VFX workflows
- CoTracker integration: [s9roll7/comfyui_cotracker_node](https://github.com/s9roll7/comfyui_cotracker_node)

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

Note: RAFT and SEA-RAFT vendor code (included in `raft_vendor/` and `searaft_vendor/`) are licensed under BSD-3-Clause. Model weights must be downloaded separately (see Installation section).
