# CoTracker Integration Guide (Pipeline B2)

This guide covers the new **Pipeline B2: CoTracker Mesh-Warp** integration for ComfyUI Motion Transfer.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Parameter Tuning](#parameter-tuning)
- [When to Use Pipeline B2](#when-to-use-pipeline-b2)
- [Troubleshooting](#troubleshooting)
- [Technical Details](#technical-details)
- [Performance Optimization](#performance-optimization)

---

## Overview

### What is CoTracker?

**CoTracker** (Meta AI, ECCV 2024) is a transformer-based point tracking model that tracks sparse points across an entire video sequence. Unlike traditional optical flow (which works frame-by-frame), CoTracker's transformer architecture sees the full video, providing superior temporal stability.

**CoTracker3** (October 2024) is the latest version, trained on 1000x less data while maintaining accuracy.

### Why Pipeline B2?

**Pipeline B2** combines CoTracker's robust point tracking with mesh-based warping for high-resolution motion transfer:

| Advantage | Description |
|-----------|-------------|
| **Temporal Stability** | Transformer sees entire video → smoother trajectories |
| **Occlusion Handling** | Tracks points through occlusions (e.g., hands in front of face) |
| **Adaptive Density** | 100-4096 points, automatically placed on high-motion areas |
| **Lower VRAM** | Sparse tracking uses 8-12GB vs 12-24GB for dense flow |
| **Organic Motion** | Excellent for faces, hands, cloth, and complex deformations |

---

## Installation

### 1. Install Motion Transfer Pack (if not already installed)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/cedarconnor/ComfyUI_MotionTransfer.git
cd ComfyUI_MotionTransfer
pip install -r requirements.txt
```

### 2. Install CoTracker Node (Required for Pipeline B2)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/s9roll7/comfyui_cotracker_node.git
```

**No additional pip install needed** - CoTracker uses torch.hub and auto-downloads models.

### 3. Restart ComfyUI

```bash
# Stop ComfyUI (Ctrl+C)
# Restart ComfyUI
```

### 4. Verify Installation

On first use, CoTracker3 model (~500MB) will auto-download from torch.hub. You'll see:

```
Loading CoTracker model: cotracker3_online
Downloading: "https://github.com/facebookresearch/co-tracker/..."
CoTracker model loaded successfully
```

---

## Quick Start

### Basic Workflow

1. **Load the example workflow:**
   - Drag `examples/workflow_pipeline_b2_cotracker.json` into ComfyUI

2. **Configure inputs:**
   ```
   LoadVideo node:
     - video: "your_video.mp4" (1080p, 3-10 seconds)

   LoadImage node:
     - image: "your_still_16k.png" (4K-16K still image)
   ```

3. **Adjust parameters (optional):**
   ```
   GridPointGeneratorNode:
     - width: 1920 (match your video)
     - height: 1080 (match your video)
     - grid_size: 64 (4096 points)

   CoTrackerNode:
     - grid_size: 64
     - max_num_of_points: 4096
     - confidence_threshold: 0.90
     - min_distance: 30

   MeshFromCoTracker:
     - frame_index: 0
     - min_triangle_area: 100.0
     - video_width: 1920
     - video_height: 1080
   ```

4. **Queue the workflow**

5. **Output:** High-res frames saved to `output/pipeline_b2_cotracker/`

---

## Parameter Tuning

### GridPointGeneratorNode

Generates a uniform grid of tracking points.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `width` | 1920 | 64-8192 | Video width (must match your video) |
| `height` | 1080 | 64-8192 | Video height (must match your video) |
| `grid_size` | 64 | 10-128 | Grid divisions (64×64 = 4096 points) |

**Recommendations:**
- **64×64 (4096 points):** Best balance for 1080p video
- **32×32 (1024 points):** Faster, lower VRAM, less detail
- **96×96 (9216 points):** More detail, higher VRAM (12GB+)

**Grid size formula:**
```
Total points = grid_size × grid_size
```

---

### CoTrackerNode

Tracks points across the video sequence.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `tracking_points` | "" | string | Optional manual points (x,y per line) |
| `grid_size` | 20 | 0-100 | Auto-grid density (0 = use manual points only) |
| `max_num_of_points` | 100 | 1-10000 | Maximum points to track |
| `confidence_threshold` | 0.90 | 0.0-1.0 | Filter unreliable tracks |
| `min_distance` | 30 | 0-500 | Minimum spacing between points (pixels) |
| `force_offload` | true | bool | Offload model after tracking (saves VRAM) |
| `enable_backward` | false | bool | Bidirectional tracking (experimental) |

**Recommendations:**

**For 1080p video:**
```json
{
  "grid_size": 64,
  "max_num_of_points": 4096,
  "confidence_threshold": 0.90,
  "min_distance": 30
}
```

**For 720p video:**
```json
{
  "grid_size": 48,
  "max_num_of_points": 2304,
  "confidence_threshold": 0.90,
  "min_distance": 25
}
```

**For fast preview (low quality):**
```json
{
  "grid_size": 32,
  "max_num_of_points": 1024,
  "confidence_threshold": 0.85,
  "min_distance": 40
}
```

**For high quality (slow):**
```json
{
  "grid_size": 96,
  "max_num_of_points": 9216,
  "confidence_threshold": 0.95,
  "min_distance": 20
}
```

**Parameter Effects:**

- **`confidence_threshold`:**
  - **0.80:** Keep more points, some may be unstable
  - **0.90:** Balanced (recommended)
  - **0.95:** Only very confident tracks, fewer points

- **`min_distance`:**
  - **10-20:** Dense mesh, more triangles, slower
  - **30:** Balanced (recommended)
  - **50-100:** Sparse mesh, faster, less detail

- **`enable_backward`:**
  - **false:** Standard forward tracking (recommended)
  - **true:** Bidirectional tracking for occlusions (experimental, slower)

---

### MeshFromCoTracker

Converts point trajectories to triangulated mesh.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `tracking_results` | - | STRING | JSON from CoTrackerNode (auto-connected) |
| `frame_index` | 0 | 0-T | Reference frame for UV coordinates |
| `min_triangle_area` | 100.0 | 1-10000 | Filter degenerate triangles (pixels²) |
| `video_width` | 1920 | 64-8192 | Original video width |
| `video_height` | 1080 | 64-8192 | Original video height |

**Recommendations:**
- **`frame_index`:** Usually 0 (first frame). Change if first frame has occlusions.
- **`min_triangle_area`:**
  - **50:** More triangles, denser mesh, slower
  - **100:** Balanced (recommended)
  - **200:** Fewer triangles, faster, less detail
- **`video_width/height`:** **Must exactly match your video resolution!**

---

## When to Use Pipeline B2

### ✅ Use Pipeline B2 When:

1. **Temporal stability is critical:**
   - Character faces (lip sync, expressions)
   - Hand gestures
   - Fabric/cloth motion
   - Any motion where flicker is unacceptable

2. **Occlusions are present:**
   - Hands in front of face
   - Objects passing in front
   - Self-occlusions (e.g., arms crossing)

3. **Large, complex deformations:**
   - Non-rigid body deformation
   - Organic shapes
   - Character animation

4. **Lower VRAM is needed:**
   - Only have 8-12GB VRAM
   - Pipeline A/B cause OOM errors

### ❌ Use Pipeline A/B Instead When:

1. **Speed is priority:**
   - Pipeline A (Flow-Warp) is 2x faster than B2

2. **Camera motion dominates:**
   - Static object, moving camera
   - Pipeline A handles this better

3. **Extreme parallax:**
   - Significant depth variation
   - Consider Pipeline C (3D-Proxy)

4. **CoTracker installation not possible:**
   - External dependency issues
   - Stick with built-in Pipeline A or B

---

## Troubleshooting

### Installation Issues

#### "CoTrackerNode not found"

**Problem:** CoTracker node not installed.

**Solution:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/s9roll7/comfyui_cotracker_node.git
# Restart ComfyUI
```

#### "Failed to load CoTracker model"

**Problem:** torch.hub download failed or model incompatible.

**Solution:**
```bash
# Check internet connection
ping github.com

# Check PyTorch version (need 1.9+)
python -c "import torch; print(torch.__version__)"

# Clear torch hub cache
rm -rf ~/.cache/torch/hub/facebookresearch_co-tracker*

# Retry in ComfyUI
```

#### "CUDA out of memory"

**Problem:** Not enough VRAM for tracking.

**Solution:**
1. Reduce `grid_size` (64 → 48 → 32)
2. Reduce `max_num_of_points` (4096 → 2048 → 1024)
3. Enable `force_offload` (should be true by default)
4. Use shorter video (3-5 seconds instead of 10)

---

### Workflow Errors

#### "No valid trajectory data found"

**Problem:** CoTrackerNode returned empty results.

**Solution:**
1. Check video is valid (not blank frames)
2. Lower `confidence_threshold` (0.90 → 0.80)
3. Increase `grid_size` for more points
4. Check video is connected to GetVideoComponents

#### "Invalid JSON in tracking_results"

**Problem:** MeshFromCoTracker received corrupted data.

**Solution:**
1. Ensure CoTrackerNode output is connected to MeshFromCoTracker input
2. Don't use "Reroute" nodes between them (can corrupt data)
3. Restart ComfyUI and reload workflow

#### "Mesh has too few triangles"

**Problem:** Not enough valid points for triangulation.

**Solution:**
1. Increase `grid_size` (more points)
2. Lower `confidence_threshold` (keep more points)
3. Reduce `min_distance` (denser mesh)
4. Lower `min_triangle_area` (keep smaller triangles)

---

### Quality Issues

#### Temporal Jitter/Flicker

**Problem:** Output has frame-to-frame inconsistency.

**Possible Causes:**
1. **Too few tracking points:**
   - Increase `grid_size` to 96 or 128
   - Lower `min_distance` to 20

2. **Unreliable tracks:**
   - Increase `confidence_threshold` to 0.95
   - Check CoTrackerNode visualization output (red dots)

3. **Video has motion blur:**
   - CoTracker struggles with blurry frames
   - Use sharper source video

#### Mesh Artifacts (Visible Triangles)

**Problem:** Triangular distortions visible in output.

**Possible Causes:**
1. **Too few points:**
   - Increase `grid_size` (64 → 96)

2. **Bad triangles:**
   - Increase `min_triangle_area` (100 → 200)

3. **Degenerate triangles:**
   - Increase `min_distance` (30 → 50)

#### Warping Inaccuracies

**Problem:** High-res output doesn't match video motion.

**Possible Causes:**
1. **Wrong video dimensions:**
   - Ensure `video_width/height` in MeshFromCoTracker exactly matches video

2. **Wrong reference frame:**
   - Try `frame_index` = 0, middle frame, or last frame

3. **Still image misaligned:**
   - Ensure still image corresponds to a frame in the video

---

## Technical Details

### Data Flow

```
Video (1080p, 120 frames)
    ↓
GridPointGeneratorNode (64×64 = 4096 points)
    ↓
CoTrackerNode
    ├─ Input: [B, T, C, H, W] video tensor
    ├─ Input: Grid points (x, y) on frame 0
    ├─ Model: CoTracker3 transformer
    └─ Output: List of JSON strings (one per point)
           Format: '[{"x":500,"y":300}, {"x":502,"y":301}, ...]'
    ↓
MeshFromCoTracker
    ├─ Parse: Newline-separated JSON strings → [T, N, 2] array
    ├─ Triangulate: Delaunay on frame 0 positions → faces
    ├─ Filter: Remove triangles with area < min_triangle_area
    └─ Output: List of mesh dicts per frame
           Format: {'vertices': [N,2], 'faces': [F,3], 'uvs': [N,2], ...}
    ↓
BarycentricWarp
    ├─ Input: Still image (16K)
    ├─ Input: Mesh sequence (from MeshFromCoTracker)
    ├─ Process: For each triangle, warp pixels via barycentric coords
    └─ Output: Warped sequence (16K, 120 frames)
    ↓
HiResWriter (save to disk)
```

---

### Mesh Format

MeshFromCoTracker outputs the same mesh format as MeshBuilder2D:

```python
mesh = {
    'vertices': np.array([N, 2], dtype=np.float32),  # Deformed positions (pixels)
    'faces': np.array([F, 3], dtype=np.int32),       # Triangle indices
    'uvs': np.array([N, 2], dtype=np.float32),       # Normalized [0,1] coords
    'width': int,                                     # Video width
    'height': int                                     # Video height
}
```

**Key Points:**
- **vertices:** Change per frame (motion)
- **faces:** Same for all frames (topology fixed)
- **uvs:** Normalized coordinates for sampling high-res still

---

### CoTracker Output Format

CoTrackerNode returns:
```python
RETURN_TYPES = ("STRING", "IMAGE")
RETURN_NAMES = ("tracking_results", "image_with_results")
```

**tracking_results format:**
```
List of JSON strings (one per point):
[
  '[{"x":500,"y":300}, {"x":502,"y":301}, {"x":504,"y":302}, ...]',
  '[{"x":200,"y":250}, {"x":201,"y":252}, {"x":203,"y":254}, ...]',
  ...
]
```

When passed to the next node, ComfyUI converts list → newline-separated string:
```
[{"x":500,"y":300}, {"x":502,"y":301}, ...]
[{"x":200,"y":250}, {"x":201,"y":252}, ...]
```

MeshFromCoTracker parses this format automatically.

---

### Delaunay Triangulation

MeshFromCoTracker uses **scipy.spatial.Delaunay** to triangulate points:

**Algorithm:**
1. Use frame 0 point positions as input
2. Delaunay triangulation in UV space (normalized [0,1])
3. Filter triangles with area < `min_triangle_area`
4. Use same topology for all frames (only vertices change)

**Why Delaunay?**
- **Maximizes minimum angle:** Avoids skinny triangles
- **Fast:** O(n log n) for 2D
- **Standard:** Same as Pipeline B (MeshBuilder2D)

---

## Performance Optimization

### Speed vs Quality Trade-offs

| Setting | Fast Preview | Balanced | High Quality |
|---------|--------------|----------|--------------|
| **grid_size** | 32 | 64 | 96 |
| **max_num_of_points** | 1024 | 4096 | 9216 |
| **confidence_threshold** | 0.85 | 0.90 | 0.95 |
| **min_distance** | 50 | 30 | 20 |
| **min_triangle_area** | 200 | 100 | 50 |
| **Processing time (1080p→16K, 120 frames)** | ~12 min | ~20 min | ~35 min |
| **VRAM usage** | 6-8GB | 8-12GB | 12-16GB |

---

### VRAM Reduction Tips

1. **Reduce grid_size:**
   ```
   64×64 = 4096 points (~10GB)
   48×48 = 2304 points (~8GB)
   32×32 = 1024 points (~6GB)
   ```

2. **Enable force_offload:**
   - Offloads CoTracker model after tracking
   - Frees ~2GB VRAM for downstream nodes

3. **Process shorter videos:**
   - CoTracker loads entire video into VRAM
   - 3 sec (72 frames) vs 10 sec (240 frames) = 3x less VRAM

4. **Reduce tile_size in BarycentricWarp:**
   - (Warping step, not tracking step)

---

### Multi-GPU Usage

CoTracker uses a single GPU. For multi-GPU setups:

1. **Option A:** Run multiple workflows in parallel
   - Each workflow on different GPU
   - Use ComfyUI's multi-queue feature

2. **Option B:** Split video into chunks
   - Process chunks sequentially on different GPUs
   - Stitch results afterward

---

## Comparison to Other Methods

### CoTracker vs RAFT Optical Flow

| Feature | RAFT (Pipeline A/B) | CoTracker (Pipeline B2) |
|---------|---------------------|-------------------------|
| **Method** | Frame-to-frame dense flow | Sparse point tracking |
| **Temporal Model** | Recurrent (per-pair) | Transformer (whole video) |
| **Occlusion Handling** | Poor (flow breaks) | Excellent (tracks through) |
| **Temporal Stability** | Good | **Excellent** |
| **Point Density** | Dense (H×W) | Sparse (100-10K points) |
| **Speed** | Fast (real-time) | Medium (~1x real-time) |
| **VRAM** | 12-24GB | 8-12GB |
| **Best For** | General motion | Faces, hands, organic |

---

### Pipeline Recommendations

| Use Case | Recommended Pipeline | Reason |
|----------|----------------------|--------|
| **General purpose** | A (Flow-Warp) | Fastest, most reliable |
| **Character faces** | **B2 (CoTracker)** | Best temporal stability |
| **Hand gestures** | **B2 (CoTracker)** | Handles occlusions |
| **Fabric/cloth** | B or **B2** | Mesh warping better than flow |
| **Camera motion** | A (Flow-Warp) | Handles parallax better |
| **Large deformations** | **B2 (CoTracker)** | Robust to extreme motion |
| **Fast preview** | A (Flow-Warp) | 2x faster than B2 |
| **Limited VRAM (8GB)** | **B2 (CoTracker)** | Lower memory footprint |

---

## References

### Papers

**CoTracker:**
- Paper: [CoTracker: It is Better to Track Together](https://arxiv.org/abs/2307.07635) (Meta AI, ECCV 2024)
- CoTracker3: [CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos](https://arxiv.org/abs/2410.11831) (October 2024)
- Project: https://co-tracker.github.io/

**RAFT (for comparison):**
- Paper: [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://arxiv.org/abs/2003.12039) (ECCV 2020)

---

### External Dependencies

- **CoTrackerNode:** https://github.com/s9roll7/comfyui_cotracker_node
- **CoTracker (Meta AI):** https://github.com/facebookresearch/co-tracker
- **scipy.spatial.Delaunay:** Included with scipy (already in requirements.txt)

---

### Credits

**CoTracker Integration:**
- Implementation: ComfyUI Motion Transfer Pack v0.2
- CoTracker: Meta AI (Nikita Karaev, Ignacio Rocco, et al.)
- CoTracker ComfyUI Node: s9roll7

**License:**
- CoTracker: Apache 2.0
- comfyui_cotracker_node: MIT (assumed, check repo)
- This integration: MIT (same as Motion Transfer Pack)

---

## Support

**Issues with Motion Transfer Pack:**
- GitHub: https://github.com/cedarconnor/ComfyUI_MotionTransfer/issues

**Issues with CoTracker Node:**
- GitHub: https://github.com/s9roll7/comfyui_cotracker_node/issues

**Issues with CoTracker Model:**
- GitHub: https://github.com/facebookresearch/co-tracker/issues

---

**Last Updated:** 2025-10-20
**Version:** Motion Transfer Pack v0.2 + CoTracker Integration
