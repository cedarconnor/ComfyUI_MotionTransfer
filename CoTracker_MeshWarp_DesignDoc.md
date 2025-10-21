# CoTracker-Driven Mesh Warp Pipeline — Technical Design Document

**Author:** ChatGPT  
**Date:** 2025-10-20  
**Purpose:** Integrate Meta’s CoTracker into the Motion Transfer (Mesh-Warp) pipeline for high-resolution image animation in ComfyUI.

---

## 🎯 Objective

Use **CoTracker** to extract temporally stable point trajectories from a low-resolution AI-generated video, then reconstruct a deforming 2D mesh and warp a **16K still image** to follow that motion.

This system forms the **Pipeline B — Mesh-Warp (CoTracker Edition)** of the broader Ultra-High-Res Motion Transfer suite.

---

## 🧩 Pipeline Overview

### Base Flow

```
[VideoFramesLoader]
      ↓
[GridPointGeneratorNode] → [CoTrackerNode]
      ↓                         ↓
                          tracker_points
                                ↓
                        [MeshFromCoTracker]
                                ↓
             vertices_ref + faces + vertices_def
                                ↓
                      [BarycentricWarpMesh]
                                ↓
                     [TemporalConsistency]
                                ↓
                         [HiResWriter]
```

### Summary of Stages
| Stage | Purpose |
|--------|----------|
| **VideoFramesLoader** | Extracts and resizes frames from the driving low-res video. |
| **GridPointGeneratorNode** | Defines evenly spaced track points for CoTracker to follow. |
| **CoTrackerNode** | Tracks those points across frames (offline or online). |
| **MeshFromCoTracker** | Builds a deformable triangular mesh from tracked point trajectories. |
| **BarycentricWarpMesh** | Warps the 16K still frame according to the mesh deformation. |
| **TemporalConsistency** | Blends neighboring frames for smooth temporal motion. |
| **HiResWriter** | Exports the high-resolution image sequence (EXR, MOV, etc.). |

---

## ⚙️ Key Components

### 1. CoTracker Node (from s9roll7/comfyui_cotracker_node)
- **Repo:** https://github.com/s9roll7/comfyui_cotracker_node  
- **Type:** Transformer-based point tracker (Meta AI)
- **Inputs:** Video frames, grid of seed points  
- **Outputs:** `tracker_points` tensor [T, N, 2] (T frames, N tracked points, XY positions)
- **Modes:**
  - *Offline:* uses full video context for best accuracy.
  - *Online:* sliding window tracking (real-time capable).
- **Useful Parameters:**
  - `grid_size`: (e.g. 64 → 4096 points)
  - `enable_backward`: tracks objects appearing mid-clip
  - `amp_x`, `amp_y`: optional motion exaggeration

---

### 2. MeshFromCoTracker (custom node)

**File:** `comfyui_motion_transfer/nodes/mesh_from_cotracker.py`  
**Purpose:** Converts CoTracker’s point trajectories into a triangular deforming mesh.

**Core Logic:**
- Frame 0 defines reference vertex positions.
- Delaunay triangulation defines consistent mesh faces.
- For each frame, vertices deform according to CoTracker output.

**Outputs:**
- `vertices_ref`: base mesh vertex positions (frame 0)
- `faces`: mesh connectivity (triangular indices)
- `vertices_def`: vertex positions at chosen frame

---

### 3. BarycentricWarpMesh (custom node)

**File:** `comfyui_motion_transfer/nodes/barycentric_warp_mesh.py`  
**Purpose:** Warps a high-resolution still image according to mesh deformation.

**Algorithm:**
- For each triangle, compute affine transform (ref → def).
- Warp region via `cv2.warpAffine()`.
- Blend warped patches with optional feather mask.

**Parameters:**
| Parameter | Default | Description |
|------------|----------|--------------|
| `blend_mode` | “none” | Use “feather” for soft seams. |
| `fill_mode` | “reflect” | Border fill mode (reflect, nearest, black). |

**Outputs:** Warped frame matching mesh motion.

---

### 4. TemporalConsistency

**Purpose:** Blend consecutive frames using flow-based warping to reduce flicker.

**Inputs:** warped frames, optional flow hints.  
**Outputs:** stabilized 16K sequence.

---

## 🧠 Motion Strategy

CoTracker yields *sparse but highly reliable* trajectories. These are triangulated into a mesh, which then defines a *piecewise-affine deformation field* applied to the 16K still image.

This ensures motion fidelity, stability, and visual realism at ultra-high resolutions.

---

## 💾 Installation Steps

```bash
# 1. Install CoTracker node
cd ComfyUI/custom_nodes
git clone https://github.com/s9roll7/comfyui_cotracker_node.git
pip install -r comfyui_cotracker_node/requirements.txt

# 2. Add custom motion-transfer nodes
git clone https://github.com/yourname/comfyui_motion_transfer.git
# (contains MeshFromCoTracker and BarycentricWarpMesh)

# 3. Restart ComfyUI
```

---

## 🧮 Example Parameter Setup

| Node | Parameter | Recommended |
|------|------------|-------------|
| CoTrackerNode | mode | offline |
|  | grid_size | 64 |
|  | window_size | 32 |
|  | amp_x / amp_y | 1.0 |
| MeshFromCoTracker | frame_index | Animated via timeline |
| BarycentricWarpMesh | blend_mode | feather |
|  | fill_mode | reflect |
| TemporalConsistency | blend_strength | 0.4 |

---

## 🧱 Output Data

| Data | Format | Description |
|-------|--------|-------------|
| Warped frames | `.exr` / `.png` | 16K high-res animation frames |
| Mesh cache | `.npz` / `.json` | Optional precomputed mesh data per frame |
| Trajectories | `.npy` | Raw CoTracker outputs |
| Preview | `.mov` (ProRes 422HQ) | 4K or 1080p playback preview |

---

## 🔬 Integration with Existing Pipelines

**For Hybrid Workflows:**
- You can combine CoTracker mesh tracking with dense flow fields (from RAFT/WAFT) to enhance small details.
- The CoTracker mesh defines large-scale deformation; optical flow adds fine micro-motion.

**Example Merge:**
```
CoTrackerMesh (macro) + OpticalFlow (micro) → Composite Warp → 16K Output
```

---

## 🧠 Performance Tips

| Task | Optimization |
|------|---------------|
| 16K processing | Use tiled warping (TileWarp16K node). |
| Memory | Precompute meshes on CPU, warp tiles on GPU. |
| Temporal smoothing | Use TemporalConsistency node or FILM interpolation. |
| Visual seams | Feather blending in BarycentricWarpMesh. |

---

## 🚀 Future Extensions

- CUDA-based `BarycentricWarpMeshTorch` using `torch.grid_sample()` for speed.
- Hybrid flow/mesh compositing node.
- Mesh export to `.obj` or `.abc` for Unreal/Nuke pipelines.
- 3D proxy generation using motion depth cues.

---

## ✅ Deliverables

| File | Purpose |
|------|----------|
| `mesh_from_cotracker.py` | Builds mesh from CoTracker trajectories |
| `barycentric_warp_mesh.py` | Warps still via mesh deformation |
| `design_cotracker_meshwarp.md` | This document |
| `examples/graph_cotracker_mesh.json` | Example ComfyUI graph |

---

## 📦 Repo Structure

```
comfyui_motion_transfer/
  nodes/
    mesh_from_cotracker.py
    barycentric_warp_mesh.py
  examples/
    graph_cotracker_mesh.json
  docs/
    design_cotracker_meshwarp.md
```

---

## 🧩 Summary

This CoTracker-Driven Mesh Warp system provides a **state-of-the-art** way to apply realistic motion from low-resolution video to ultra-high-resolution stills.

It offers:
- Temporal stability (transformer tracking)
- Non-rigid surface fidelity (mesh warping)
- Modular integration in ComfyUI

Perfect for:
- Stylized AI cinematics
- Projection-mapping workflows
- VFX motion retargeting
- Still-to-video animation at 8K–16K

---
