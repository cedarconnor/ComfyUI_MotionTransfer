# CoTracker Integration Plan for ComfyUI Motion Transfer

**Based on:** `CoTracker_MeshWarp_DesignDoc.md`
**Created:** 2025-01-20
**Status:** READY FOR IMPLEMENTATION

---

## Executive Summary

### What is CoTracker?

**CoTracker** (Meta AI, ECCV 2024) is a transformer-based point tracking model that tracks dense points jointly across video frames, accounting for their dependencies. **CoTracker3** (October 2024) is the latest version with simplified architecture and 1000x less training data.

**Key Advantages over RAFT Flow:**
- ✅ **Temporally stable** - Transformer tracks points across entire video
- ✅ **Occlusion-aware** - Can track points that go behind objects
- ✅ **Sparse but accurate** - 4K-70K points (vs 2M pixels with flow)
- ✅ **Better for mesh warping** - Provides clean trajectories for Delaunay triangulation
- ✅ **Memory efficient** - Uses token proxies for joint tracking

**When to Use:**
- ✅ Large deformations (character animation, fabric, organic motion)
- ✅ Objects moving in/out of view
- ✅ Long sequences requiring temporal consistency
- ✅ Mesh-based warping preferred over pixel-based

---

## Current State Analysis

### What We Already Have ✅

**Existing Pipeline B (RAFT-based Mesh-Warp):**
```
RAFTFlowExtractor → MeshBuilder2D → AdaptiveTessellate → BarycentricWarp
```

**Components:**
1. ✅ `MeshBuilder2D` - Creates mesh from optical flow
2. ✅ `AdaptiveTessellate` - Refines mesh based on flow gradients
3. ✅ `BarycentricWarp` - Warps high-res image using mesh
4. ✅ `TemporalConsistency` - Frame blending for smooth output
5. ✅ `HiResWriter` - Exports high-res sequences

**What Works:**
- Mesh generation from dense flow fields
- Delaunay triangulation
- Barycentric interpolation warping
- High-res tiled processing

### What's Missing for CoTracker ⚠️

1. ❌ **CoTracker node integration** - Need to install `s9roll7/comfyui_cotracker_node`
2. ❌ **`MeshFromCoTracker`** - Convert point trajectories → mesh
3. ❌ **Adapter logic** - CoTracker outputs `[T, N, 2]` vs RAFT `[B, H, W, 2]`
4. ❌ **Example workflow** - Pipeline B with CoTracker variant

---

## Integration Strategy

### Approach: **Parallel Pipeline, Reuse Existing Nodes**

**Option A: Separate CoTracker Pipeline** ✅ **RECOMMENDED**
- Keep existing RAFT-based Pipeline B unchanged
- Create new Pipeline B2 (CoTracker variant)
- Reuse `BarycentricWarp`, `TemporalConsistency`, `HiResWriter`
- Add new `MeshFromCoTracker` node only

**Why:**
- ✅ No breaking changes to existing workflows
- ✅ Users can choose RAFT or CoTracker
- ✅ Minimal code changes (1 new node)
- ✅ Easy to compare results

**Option B: Replace MeshBuilder2D** ❌ NOT RECOMMENDED
- Would break existing workflows
- Less flexible for users

---

## Implementation Plan

### Phase 1: Dependencies & Setup (2-3 hours)

**Goal:** Install CoTracker node and verify it works

**Tasks:**
1. Install s9roll7's CoTracker node
2. Test basic point tracking
3. Understand output format `[T, N, 2]`
4. Document integration points

**Commands:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/s9roll7/comfyui_cotracker_node.git
pip install -r comfyui_cotracker_node/requirements.txt
```

**Testing:**
- Load test video in ComfyUI
- Use GridPointGeneratorNode (64x64 = 4096 points)
- Run CoTrackerNode (offline mode)
- Verify output shape: `[T, 4096, 2]`

**Deliverables:**
- CoTracker node installed and working
- Sample trajectory data saved

---

### Phase 2: Create MeshFromCoTracker Node (4-6 hours)

**Goal:** Convert CoTracker trajectories into mesh format compatible with `BarycentricWarp`

**File:** `motion_transfer_nodes.py` (add new class)

**Node Specification:**

```python
class MeshFromCoTracker:
    """Build deformation mesh from CoTracker point trajectories.

    Converts sparse point tracks [T, N, 2] into triangulated mesh sequence
    compatible with BarycentricWarp node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tracker_points": ("COTRACKER_POINTS", {
                    "tooltip": "Point trajectories from CoTrackerNode. Shape: [T, N, 2] where T=frames, N=points, 2=XY coordinates."
                }),
                "frame_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Which frame to use as reference (usually 0). Deformation is relative to this frame."
                }),
                "min_triangle_area": ("FLOAT", {
                    "default": 100.0,
                    "min": 1.0,
                    "max": 10000.0,
                    "tooltip": "Filter out tiny/degenerate triangles. Same as MeshBuilder2D parameter."
                }),
                "video_width": ("INT", {
                    "default": 1920,
                    "tooltip": "Original video width (for UV normalization)"
                }),
                "video_height": ("INT", {
                    "default": 1080,
                    "tooltip": "Original video height (for UV normalization)"
                }),
            }
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh_sequence",)
    FUNCTION = "build_mesh_from_tracks"
    CATEGORY = "MotionTransfer/Mesh"
```

**Core Algorithm:**

```python
def build_mesh_from_tracks(self, tracker_points, frame_index, min_triangle_area, video_width, video_height):
    """Convert CoTracker trajectories to mesh sequence.

    Args:
        tracker_points: [T, N, 2] array of XY positions per frame
        frame_index: Reference frame (0)
        min_triangle_area: Filter threshold
        video_width, video_height: Original video dimensions

    Returns:
        mesh_sequence: List of mesh dicts (same format as MeshBuilder2D)
    """
    from scipy.spatial import Delaunay

    # tracker_points shape: [T, N, 2]
    T, N, _ = tracker_points.shape

    # Reference frame (frame 0)
    ref_points = tracker_points[frame_index]  # [N, 2]

    # Build UVs from reference positions (normalized [0, 1])
    uvs = np.array([
        [x / video_width, y / video_height]
        for x, y in ref_points
    ])

    # Delaunay triangulation on reference frame
    tri = Delaunay(uvs)
    faces_base = tri.simplices

    # Filter degenerate triangles
    valid_faces = []
    for face in faces_base:
        v0, v1, v2 = ref_points[face]
        area = 0.5 * abs((v1[0] - v0[0]) * (v2[1] - v0[1]) -
                        (v2[0] - v0[0]) * (v1[1] - v0[1]))
        if area >= min_triangle_area:
            valid_faces.append(face)

    faces = np.array(valid_faces, dtype=np.int32)

    # Build mesh for each frame
    meshes = []
    for t in range(T):
        vertices = tracker_points[t]  # [N, 2] - deformed positions

        mesh = {
            'vertices': vertices.astype(np.float32),
            'faces': faces,
            'uvs': uvs.astype(np.float32),
            'width': video_width,
            'height': video_height
        }
        meshes.append(mesh)

    return (meshes,)
```

**Key Differences from MeshBuilder2D:**
- Input is sparse points `[T, N, 2]` not dense flow `[B, H, W, 2]`
- Triangulation done once on reference frame (not per-frame)
- Vertices directly from tracks (no flow displacement needed)
- More stable across time (CoTracker ensures temporal consistency)

**Deliverables:**
- `MeshFromCoTracker` class added to `motion_transfer_nodes.py`
- Unit test with sample CoTracker data
- Verified output format matches `BarycentricWarp` input

---

### Phase 3: Create Example Workflow (2-3 hours)

**Goal:** Demonstrate complete Pipeline B2 (CoTracker variant)

**Workflow File:** `examples/workflow_pipeline_b2_cotracker.json`

**Node Graph:**
```
LoadVideo (1080p driving video)
    ↓
GetVideoComponents
    ↓
GridPointGeneratorNode (grid_size=64 → 4096 points)
    ↓
CoTrackerNode (mode=offline, window_size=32)
    ↓ tracker_points [T, 4096, 2]
    ↓
MeshFromCoTracker (frame_index=0, min_triangle_area=100.0)
    ↓ mesh_sequence
    ↓
LoadImage (16K still)
    ↓
BarycentricWarp (interpolation=linear, still + mesh_sequence)
    ↓
TemporalConsistency (blend_strength=0.4)
    ↓
HiResWriter (output/pipeline_b2_cotracker/frame, format=png)
```

**Parameter Recommendations:**

| Node | Parameter | Value | Rationale |
|------|-----------|-------|-----------|
| GridPointGeneratorNode | grid_size | 64 | 4096 points = good balance |
| CoTrackerNode | mode | offline | Best accuracy (uses full video context) |
|  | window_size | 32 | Good temporal window |
|  | enable_backward | true | Track objects appearing mid-clip |
| MeshFromCoTracker | frame_index | 0 | First frame as reference |
|  | min_triangle_area | 100.0 | Filter degenerate triangles |
| BarycentricWarp | interpolation | linear | Faster, mesh already smooth |
| TemporalConsistency | blend_strength | 0.4 | Medium smoothing |

**Deliverables:**
- Complete workflow JSON
- Test with sample video + 16K still
- Verify output quality

---

### Phase 4: Documentation (2-3 hours)

**Goal:** Update all documentation with CoTracker integration

**Files to Update:**

**1. README.md**
- Add Pipeline B2 section (CoTracker variant)
- Comparison table: RAFT vs CoTracker
- Installation instructions for CoTracker node

**2. IMPLEMENTATION_PLAN.md**
- Mark CoTracker integration as complete
- Update node count (13 nodes total)

**3. examples/README.md**
- Add workflow_pipeline_b2_cotracker.json entry
- Usage guide for CoTracker parameters

**4. New: COTRACKER_GUIDE.md**
- When to use CoTracker vs RAFT
- CoTracker parameter tuning guide
- Performance benchmarks
- Troubleshooting

**Deliverables:**
- All docs updated
- Clear guidance on choosing RAFT vs CoTracker

---

## Timeline & Effort

| Phase | Tasks | Hours | Priority |
|-------|-------|-------|----------|
| **Phase 1: Setup** | Install CoTracker, test basic tracking | 2-3h | HIGH |
| **Phase 2: MeshFromCoTracker** | Implement conversion node | 4-6h | HIGH |
| **Phase 3: Workflow** | Create example, test end-to-end | 2-3h | HIGH |
| **Phase 4: Documentation** | Update all docs | 2-3h | MEDIUM |
| **TOTAL** | | **10-15 hours** | |

**Recommended Approach:**
1. Start with Phase 1+2 (6-9h) - Get core functionality working
2. Test with real data, gather feedback
3. Complete Phase 3+4 (4-6h) based on learnings

---

## Technical Comparison: RAFT vs CoTracker

| Aspect | RAFT Flow | CoTracker |
|--------|-----------|-----------|
| **Output** | Dense flow [H, W, 2] | Sparse points [T, N, 2] |
| **Points Tracked** | ~2M pixels (1080p) | 4K-70K points |
| **Temporal Stability** | Frame-to-frame only | Transformer (full video context) |
| **Occlusion Handling** | Flow stops | Tracks through occlusions |
| **Memory Usage** | High (dense field) | Low (sparse points) |
| **Speed** | Fast (2-7 sec/frame with SEA-RAFT) | Medium (depends on # points) |
| **Best For** | General motion, fine details | Large deformations, organic motion |
| **Mesh Quality** | Good (needs filtering) | Excellent (clean trajectories) |

---

## Integration Points

### Data Format Compatibility

**MeshBuilder2D (RAFT) Input:**
```python
flow: [B, H, W, 2]  # Dense displacement field
# Example: [120, 1080, 1920, 2]
```

**MeshFromCoTracker Input:**
```python
tracker_points: [T, N, 2]  # Sparse point trajectories
# Example: [120, 4096, 2]  # 64x64 grid = 4096 points
```

**BarycentricWarp Input (Common):**
```python
mesh_sequence: List[Dict]
# Each dict: {'vertices': [N, 2], 'faces': [F, 3], 'uvs': [N, 2], 'width': int, 'height': int}
```

**✅ Solution:** Both pipelines output same mesh format, so `BarycentricWarp` works with either.

---

## Performance Expectations

**Hardware:** RTX 4090, 24GB VRAM
**Input:** 1080p video (5 sec @ 24fps = 120 frames) → 16K still

| Pipeline | Tracking Time | Mesh Build | Warping | Total | Notes |
|----------|--------------|------------|---------|-------|-------|
| **B (RAFT)** | ~14 min | ~30 sec | ~14 min | ~29 min | Dense flow, all pixels |
| **B2 (CoTracker)** | ~5 min | ~10 sec | ~14 min | ~20 min | Sparse points, faster |

**Speedup: ~1.4x faster with CoTracker** (assuming 4096 tracking points)

**Quality Differences:**
- CoTracker: Better temporal stability, cleaner mesh
- RAFT: More fine-grained detail, better for small motions

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **CoTracker node API changes** | Low | High | Pin to specific version, test thoroughly |
| **Data format mismatch** | Medium | Medium | Clear documentation, validation checks |
| **Performance slower than expected** | Medium | Low | Reduce grid_size (64→32), use online mode |
| **Mesh quality issues** | Low | Medium | Filter degenerate triangles, tune min_area |
| **User confusion (too many options)** | Medium | Low | Clear docs, comparison table |

**Overall Risk:** **LOW** - Clean integration, reuses existing nodes

---

## Success Metrics

**Technical:**
- [ ] CoTracker node installed and working
- [ ] MeshFromCoTracker outputs valid mesh format
- [ ] BarycentricWarp accepts CoTracker meshes
- [ ] End-to-end pipeline produces 16K output
- [ ] Performance within 2x of RAFT pipeline

**User Experience:**
- [ ] Clear installation instructions
- [ ] Example workflow loads and runs
- [ ] Documentation explains when to use CoTracker vs RAFT
- [ ] No breaking changes to existing workflows

**Quality:**
- [ ] Mesh is temporally stable
- [ ] No visible seams or artifacts
- [ ] Better than RAFT for large deformations
- [ ] Output is production-ready

---

## Future Enhancements (Post-Initial Release)

### Phase 5: Optimizations (Optional)
- **CUDA-accelerated BarycentricWarp** using `torch.grid_sample()`
- **Tiled CoTracker processing** for very long videos
- **Adaptive grid density** (more points in high-motion areas)

### Phase 6: Hybrid Workflows (Optional)
- **CoTracker (macro) + RAFT (micro)** - Combine for best of both
- **Flow-guided mesh refinement** - Use flow to subdivide CoTracker mesh
- **Mesh caching** - Precompute meshes, save to disk

### Phase 7: Export Features (Optional)
- **Mesh export to Alembic (.abc)** for Nuke/Houdini
- **Mesh export to OBJ sequence** for 3D software
- **Trajectory visualization** - Debug view of tracked points

---

## Alternative Approaches Considered

### Option A: Modify MeshBuilder2D to Accept CoTracker ❌
**Pros:** Single node for both
**Cons:** Complex logic, confusing UI, breaks existing workflows
**Decision:** Rejected - Too complex

### Option B: Separate CoTracker Package ❌
**Pros:** Clean separation
**Cons:** More repos to maintain, confusing for users
**Decision:** Rejected - Keep it simple

### Option C: Add MeshFromCoTracker (Separate Node) ✅ **CHOSEN**
**Pros:** Clean, reuses existing nodes, no breaking changes
**Cons:** Slightly more nodes in menu
**Decision:** **Accepted** - Best balance

---

## Dependencies

**Required:**
- s9roll7/comfyui_cotracker_node (external)
- scipy (already in requirements.txt)
- numpy (already in requirements.txt)

**Optional:**
- CoTracker3 models (auto-download from HuggingFace)

---

## Installation Instructions (Final)

**1. Install Base Package (if not done):**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/cedarconnor/ComfyUI_MotionTransfer.git
cd ComfyUI_MotionTransfer
pip install -r requirements.txt
```

**2. Install CoTracker Node:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/s9roll7/comfyui_cotracker_node.git
pip install -r comfyui_cotracker_node/requirements.txt
```

**3. Install SEA-RAFT or RAFT (optional, for Pipeline A):**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/princeton-vl/SEA-RAFT.git
```

**4. Restart ComfyUI**

---

## Open Questions

1. **Grid density:** What grid_size works best for different video resolutions?
   - **Recommendation:** Start with 64 (4096 points), test 32/48/96

2. **Memory limits:** How many points can CoTracker track on different GPUs?
   - **Research needed:** Test on 8GB, 12GB, 24GB VRAM

3. **Offline vs Online mode:** When should users prefer online tracking?
   - **Recommendation:** Offline for best quality, online for real-time preview

4. **Mesh filtering:** Same min_triangle_area as MeshBuilder2D?
   - **Recommendation:** Yes, 100.0 is good default

---

## Next Steps

**Immediate (< 1 week):**
1. ✅ Create this implementation plan
2. Install and test s9roll7's CoTracker node
3. Implement `MeshFromCoTracker` node
4. Test with sample data

**Short-term (1-2 weeks):**
5. Create example workflow
6. Test end-to-end pipeline
7. Gather performance benchmarks

**Medium-term (2-4 weeks):**
8. Update documentation
9. Create COTRACKER_GUIDE.md
10. Git commit and push

**Before Release:**
11. Test on multiple GPUs (8GB, 12GB, 24GB)
12. Verify with different video types (character, fabric, organic)
13. Compare quality vs RAFT pipeline

---

## Conclusion

**CoTracker integration is HIGHLY RECOMMENDED:**

✅ **Pros:**
- State-of-the-art point tracking (ECCV 2024)
- Better temporal stability than RAFT
- Excellent for mesh-based warping
- Clean integration with existing nodes
- Faster processing (sparse vs dense)
- Handles occlusions gracefully

⚠️ **Cons:**
- Requires additional node installation
- Slightly more complex setup
- May be overkill for simple motion

**Effort:** 10-15 hours total
**Impact:** High - Significant quality improvement for Pipeline B
**Risk:** Low - Clean integration, no breaking changes

**Recommendation:** **Proceed with implementation** following the 4-phase plan above.

---

## References

- **CoTracker:** https://github.com/facebookresearch/co-tracker
- **CoTracker Paper:** https://arxiv.org/abs/2307.07635
- **CoTracker3:** https://arxiv.org/abs/2410.11831
- **ComfyUI CoTracker Node:** https://github.com/s9roll7/comfyui_cotracker_node
- **Design Doc:** `CoTracker_MeshWarp_DesignDoc.md`

---

*Plan created: 2025-01-20*
*Status: READY FOR IMPLEMENTATION*
*Next: Phase 1 - Install and test CoTracker node*
