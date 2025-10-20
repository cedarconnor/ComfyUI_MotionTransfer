# SEA-RAFT Integration Plan

Plan for integrating SEA-RAFT (Simple, Efficient, Accurate RAFT) into ComfyUI Motion Transfer.

---

## Executive Summary

**SEA-RAFT** is a state-of-the-art optical flow model (ECCV 2024 Oral, Best Paper Award Candidate) that offers:
- **22.9% better accuracy** than original RAFT on Spring benchmark
- **2.3x faster** inference speed
- **Simpler architecture** using standard ResNet backbones
- **Better generalization** via rigid-motion pre-training
- **HuggingFace integration** for easy model loading

### Recommendation
✅ **Implement SEA-RAFT as alternative to RAFT** - Offers significant speed and accuracy improvements with minimal integration effort.

---

## Key Technical Differences

### Architecture Changes (SEA-RAFT vs RAFT)

| Component | Original RAFT | SEA-RAFT | Impact |
|-----------|--------------|----------|--------|
| **Feature Encoder** | Custom CNN layers | Truncated ResNet (ImageNet pretrained) | Easier training, better features |
| **Context Encoder** | Custom CNN layers | Truncated ResNet (ImageNet pretrained) | Simpler architecture |
| **Update Operator** | ConvGRU | 2x ConvNeXt blocks | Faster, more stable training |
| **Initial Flow** | Zeros | Regressed from correlation | Faster convergence (fewer iters needed) |
| **Loss Function** | L1 + EPE | Mixture of Laplace | Better accuracy |
| **Pre-training** | Standard datasets | Rigid-motion pre-training | Better generalization |

### Model Variants

| Model | Backbone | Inference Iters | Speed | Accuracy | Use Case |
|-------|----------|----------------|-------|----------|----------|
| **SEA-RAFT(S)** | ResNet-18 (6 layers) | 6 | Fastest | Good | Real-time, 8GB VRAM |
| **SEA-RAFT(M)** | ResNet-34 (13 layers) | 12 | Balanced | Better | Recommended (12-24GB VRAM) |
| **SEA-RAFT(L)** | ResNet-50 (full) | 24 | Slowest | Best | Highest quality (24GB+ VRAM) |

### Performance Benchmarks

**Spring Benchmark (SEA-RAFT vs original RAFT):**
- EPE: 3.69 vs 4.79 (22.9% improvement)
- 1px outlier: 0.36 vs 0.44 (17.8% improvement)
- Speed: 2.3x faster at same quality level

**For Motion Transfer Use Case:**
- **1080p video → 16K still**: ~3-5 seconds per frame (vs 7 seconds with RAFT)
- **Fewer iterations needed**: 6-12 iters (vs 12-20 with RAFT) for same accuracy
- **Better edge preservation**: Important for high-res warping

---

## Integration Issues & Solutions

### Issue 1: Different Model Loading API ⚠️ MODERATE

**Problem:**
- Current code uses manual checkpoint loading: `torch.load()` + `model.load_state_dict()`
- SEA-RAFT supports both local checkpoints AND HuggingFace Hub loading
- Model architecture initialization is different (ResNet-based)

**Solution:**
```python
# Current RAFT approach
from raft import RAFT
args = argparse.Namespace(small=False, ...)
model = RAFT(args)
model.load_state_dict(torch.load(checkpoint_path))

# SEA-RAFT approach (2 options)

# Option A: Local checkpoint (similar to current)
from sea_raft import SEARAFT
model = SEARAFT.load_from_checkpoint('models/sea-raft/spring-M.pth')

# Option B: HuggingFace Hub (recommended)
from sea_raft import SEARAFT
model = SEARAFT.from_pretrained('MemorySlices/Tartan-C-T-TSKH-spring540x960-M')
```

**Implementation Strategy:**
1. Add SEA-RAFT as optional dependency in requirements.txt
2. Extend `RAFTFlowExtractor` to support both RAFT and SEA-RAFT
3. Auto-detect model type from model_name parameter
4. Use HuggingFace Hub for easy downloads (no manual checkpoint management)

**Effort**: ~4 hours (modify model loading, add HF integration)

---

### Issue 2: Model Installation & Dependencies ⚠️ LOW

**Problem:**
- SEA-RAFT requires PyTorch 2.2.0+ (current code likely uses older versions)
- Requires `huggingface-hub` for model downloads
- No pip package available (must clone repo)

**Solution:**
```bash
# Add to requirements.txt
torch>=2.2.0
huggingface-hub>=0.20.0

# Installation instructions (README update)
# Option A: Clone SEA-RAFT repo
git clone https://github.com/princeton-vl/SEA-RAFT
cd SEA-RAFT && pip install -e .

# Option B: Add SEA-RAFT to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/SEA-RAFT"
```

**Implementation Strategy:**
1. Update requirements.txt with optional dependencies: `sea-raft @ git+https://github.com/princeton-vl/SEA-RAFT.git`
2. Add installation guide to README
3. Add graceful fallback: if SEA-RAFT not available, hide SEA-RAFT options in UI

**Effort**: ~2 hours (documentation, dependency management)

---

### Issue 3: Model Checkpoint Management ⚠️ LOW

**Problem:**
- Current code expects checkpoints in `models/raft/*.pth`
- SEA-RAFT has different checkpoint format and naming
- HuggingFace models auto-download to cache directory

**Solution:**
```python
# Use HuggingFace Hub caching (automatic downloads)
from huggingface_hub import hf_hub_download

model_map = {
    "sea-raft-small": "MemorySlices/SEA-RAFT-S",
    "sea-raft-medium": "MemorySlices/Tartan-C-T-TSKH-spring540x960-M",
    "sea-raft-large": "MemorySlices/SEA-RAFT-L",
}

# Downloads to ~/.cache/huggingface automatically
checkpoint = hf_hub_download(repo_id=model_map[model_name], filename="model.pth")
```

**Implementation Strategy:**
1. Support both local checkpoints (current) and HuggingFace Hub (new)
2. Add `use_huggingface` parameter to node (default: True for SEA-RAFT)
3. First run downloads model, subsequent runs use cache
4. Add progress bar for model downloads

**Effort**: ~3 hours (HF integration, caching, UI feedback)

---

### Issue 4: API Compatibility (Inference) ✅ MINIMAL

**Problem:**
- Need to ensure SEA-RAFT returns same output format as RAFT
- Current code expects: `flow [B-1, H, W, 2]` and `confidence [B-1, H, W, 1]`

**Solution:**
SEA-RAFT API is nearly identical to RAFT:

```python
# RAFT API (current)
flow_low, flow_up = model(img1, img2, iters=12, test_mode=True)
# Returns: flow_up [1, 2, H, W]

# SEA-RAFT API (same!)
flow_low, flow_up = model(img1, img2, iters=12, test_mode=True)
# Returns: flow_up [1, 2, H, W]
```

**Key Differences:**
1. SEA-RAFT **also returns uncertainty** (optional third output)
2. SEA-RAFT needs **fewer iterations** (6-12 vs 12-20) for same quality

**Implementation Strategy:**
1. Use same inference code for both models
2. If SEA-RAFT detected, reduce default iterations from 12 → 8
3. Optionally expose SEA-RAFT's uncertainty output as confidence (better than current heuristic)

**Effort**: ~1 hour (minimal changes, mostly config)

---

### Issue 5: Model Selection UI ⚠️ LOW

**Problem:**
- Current UI shows: `["raft-things", "raft-sintel", "raft-small"]`
- Need to add SEA-RAFT variants without confusing users

**Solution:**
```python
# Proposed new model selection
model_options = [
    # Original RAFT models
    "raft-things",
    "raft-sintel",
    "raft-small",
    # SEA-RAFT models (new)
    "sea-raft-small",   # Fast, 8GB VRAM
    "sea-raft-medium",  # Recommended, 12-24GB VRAM
    "sea-raft-large",   # Best quality, 24GB+ VRAM
]

# Update tooltip
"tooltip": "Optical flow model. RAFT: original (2020). SEA-RAFT: newer, 2.3x faster with 22% better accuracy (ECCV 2024). Recommended: 'sea-raft-medium' for best speed/quality balance."
```

**Implementation Strategy:**
1. Add SEA-RAFT options to dropdown
2. Update tooltip to explain differences
3. Set `sea-raft-medium` as new default (but keep RAFT for compatibility)
4. Add warning if SEA-RAFT selected but not installed

**Effort**: ~2 hours (UI updates, tooltips, validation)

---

### Issue 6: License Compatibility ✅ NO ISSUE

**Status:** Both RAFT and SEA-RAFT use **BSD-3-Clause license**

**Compatibility:**
- ✅ Compatible with MIT (ComfyUI's license)
- ✅ Compatible with commercial use
- ✅ Can bundle/redistribute with attribution
- ✅ No GPL conflicts

**Implementation Strategy:**
1. Add SEA-RAFT attribution to README
2. Include LICENSE reference in documentation
3. No code changes needed

**Effort**: ~0.5 hours (documentation only)

---

## Implementation Plan

### Phase 1: Foundation (4-6 hours)

**Goal:** Get SEA-RAFT loading and running alongside RAFT

**Tasks:**
1. ✅ Research SEA-RAFT architecture and API
2. Add SEA-RAFT to requirements.txt (optional dependency)
3. Update installation guide in README
4. Create `_load_searaft_model()` helper function
5. Test model loading with HuggingFace Hub
6. Verify inference output format matches RAFT

**Deliverables:**
- SEA-RAFT models can load and run inference
- Outputs are compatible with rest of pipeline
- Documentation updated

**Testing:**
- Load sea-raft-medium from HuggingFace
- Run on 2 test frames (1080p)
- Verify flow shape: [1, 2, H, W]
- Compare output quality vs RAFT visually

---

### Phase 2: Integration (6-8 hours)

**Goal:** Seamlessly integrate SEA-RAFT into existing RAFTFlowExtractor node

**Tasks:**
1. Extend `_load_model()` to detect RAFT vs SEA-RAFT from model_name
2. Add model_name options: `sea-raft-small`, `sea-raft-medium`, `sea-raft-large`
3. Implement HuggingFace Hub download with progress feedback
4. Add graceful fallback if SEA-RAFT not installed
5. Update tooltip to explain RAFT vs SEA-RAFT differences
6. Set default to `sea-raft-medium` (with note about RAFT compatibility)
7. Add iteration count auto-adjustment (8 for SEA-RAFT vs 12 for RAFT)

**Code Changes:**
```python
# motion_transfer_nodes.py - RAFTFlowExtractor

@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "images": ("IMAGE", {"tooltip": "..."}),
            "raft_iters": ("INT", {
                "default": 12,  # Will auto-adjust based on model
                "min": 6,
                "max": 32,
                "tooltip": "Refinement iterations. SEA-RAFT needs fewer (6-8) than RAFT (12-20) for same quality."
            }),
            "model_name": ([
                "raft-things",
                "raft-sintel",
                "raft-small",
                "sea-raft-small",
                "sea-raft-medium",  # NEW DEFAULT
                "sea-raft-large",
            ], {
                "default": "sea-raft-medium",
                "tooltip": "Flow model. RAFT: original. SEA-RAFT: 2.3x faster, 22% more accurate (ECCV 2024). Recommended: sea-raft-medium."
            }),
        }
    }

@classmethod
def _load_model(cls, model_name, device):
    """Load RAFT or SEA-RAFT model with caching."""
    if cls._model is None or cls._model_path != model_name:

        # Detect model type
        is_searaft = model_name.startswith("sea-raft")

        if is_searaft:
            # Load SEA-RAFT
            try:
                from sea_raft import SEARAFT
                from huggingface_hub import hf_hub_download
            except ImportError:
                raise ImportError(
                    "SEA-RAFT not found. Install with:\n"
                    "pip install git+https://github.com/princeton-vl/SEA-RAFT.git\n"
                    "pip install huggingface-hub"
                )

            # Map model names to HuggingFace repos
            hf_models = {
                "sea-raft-small": "MemorySlices/SEA-RAFT-S",
                "sea-raft-medium": "MemorySlices/Tartan-C-T-TSKH-spring540x960-M",
                "sea-raft-large": "MemorySlices/SEA-RAFT-L",
            }

            repo_id = hf_models[model_name]
            print(f"Downloading SEA-RAFT model from HuggingFace: {repo_id}")

            # Load from HuggingFace Hub (auto-caches)
            cls._model = SEARAFT.from_pretrained(repo_id)
            cls._model = cls._model.to(device).eval()
            cls._model_path = model_name

        else:
            # Load original RAFT (existing code)
            try:
                import sys
                sys.path.append('path/to/RAFT/core')
                from raft import RAFT
                import argparse
            except ImportError:
                raise ImportError("RAFT not found...")

            # ... existing RAFT loading code ...

    return cls._model
```

**Deliverables:**
- SEA-RAFT models appear in dropdown
- Models download automatically from HuggingFace
- Existing workflows continue to work with RAFT
- New workflows can use SEA-RAFT for better performance

**Testing:**
- Create test workflow with sea-raft-medium
- Process 10-frame 1080p video → 16K still
- Measure speed improvement (should be ~2x faster)
- Verify output quality (should be better edge preservation)
- Test fallback if SEA-RAFT not installed

---

### Phase 3: Optimization (4-6 hours)

**Goal:** Maximize SEA-RAFT benefits and improve UX

**Tasks:**
1. Add uncertainty output as improved confidence map
2. Auto-tune iteration count based on model type
3. Add model download progress bar (HuggingFace)
4. Create SEA-RAFT example workflow
5. Add performance benchmarks to README
6. Update TOOLTIPS.md with SEA-RAFT guidance

**Enhancements:**
```python
# Expose SEA-RAFT's native uncertainty as confidence
def extract_flow(self, images, raft_iters, model_name):
    # ...

    is_searaft = model_name.startswith("sea-raft")

    # Auto-adjust iterations for SEA-RAFT
    if is_searaft and raft_iters == 12:  # User didn't change default
        raft_iters = 8  # SEA-RAFT needs fewer iterations

    # Run inference
    if is_searaft:
        flow_low, flow_up, uncertainty = model(img1, img2, iters=raft_iters, test_mode=True)
        # Use native uncertainty instead of heuristic
        conf = 1.0 - uncertainty  # Convert to confidence
    else:
        flow_low, flow_up = model(img1, img2, iters=raft_iters, test_mode=True)
        # Use heuristic confidence (existing code)
        flow_mag = torch.sqrt(flow_up[:, 0:1]**2 + flow_up[:, 1:2]**2)
        conf = torch.exp(-flow_mag / 10.0)
```

**Deliverables:**
- Better confidence maps from SEA-RAFT's native uncertainty
- Automatic performance tuning
- User-friendly model download experience
- Complete documentation

**Testing:**
- Compare confidence maps: SEA-RAFT uncertainty vs heuristic
- Verify auto-iteration tuning improves speed
- Test workflow examples with SEA-RAFT
- Benchmark full pipeline speed improvement

---

### Phase 4: Polish & Documentation (2-3 hours)

**Goal:** Production-ready SEA-RAFT integration

**Tasks:**
1. Add SEA-RAFT section to README
2. Create comparison chart (RAFT vs SEA-RAFT)
3. Update example workflows to use sea-raft-medium
4. Add troubleshooting guide
5. Update IMPLEMENTATION_PLAN.md
6. Add SEA-RAFT attribution and license info

**Documentation Updates:**

**README.md additions:**
```markdown
## Model Selection: RAFT vs SEA-RAFT

### SEA-RAFT (Recommended) ⭐

**Advantages:**
- 2.3x faster inference
- 22% more accurate (ECCV 2024 Best Paper Candidate)
- Better edge preservation for high-res warping
- Auto-downloads from HuggingFace (no manual setup)

**Models:**
- `sea-raft-small`: Fast, 8GB VRAM, ~3 sec/frame @ 1080p→16K
- `sea-raft-medium`: Balanced, 12-24GB VRAM, ~4 sec/frame (recommended)
- `sea-raft-large`: Best quality, 24GB+ VRAM, ~6 sec/frame

### Original RAFT

**Use when:**
- Compatibility with existing workflows
- Already have RAFT checkpoints downloaded
- Using older PyTorch versions (< 2.2.0)

**Models:**
- `raft-things`: Trained on synthetic data
- `raft-sintel`: Best for natural video (recommended for RAFT)
- `raft-small`: Faster but less accurate

### Installation

**For SEA-RAFT (recommended):**
```bash
pip install git+https://github.com/princeton-vl/SEA-RAFT.git
pip install huggingface-hub
```

Models download automatically on first use.

**For original RAFT:**
```bash
pip install git+https://github.com/princeton-vl/RAFT.git
# Download checkpoints manually from repo
```
```

**Deliverables:**
- Complete documentation
- Example workflows with SEA-RAFT
- Troubleshooting guide
- License/attribution

---

## Timeline & Effort Estimate

| Phase | Tasks | Hours | Priority |
|-------|-------|-------|----------|
| **Phase 1: Foundation** | Research, dependencies, basic loading | 4-6h | HIGH |
| **Phase 2: Integration** | Node updates, UI, model selection | 6-8h | HIGH |
| **Phase 3: Optimization** | Uncertainty, auto-tuning, UX polish | 4-6h | MEDIUM |
| **Phase 4: Documentation** | README, examples, troubleshooting | 2-3h | MEDIUM |
| **TOTAL** | | **16-23 hours** | |

**Recommended Approach:**
1. Start with Phase 1+2 (10-14h) - Gets SEA-RAFT working
2. Release as beta, gather feedback
3. Complete Phase 3+4 (6-9h) based on user needs

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **SEA-RAFT installation issues** | Medium | High | Provide fallback to RAFT, clear error messages |
| **HuggingFace download failures** | Low | Medium | Cache models, support local checkpoints |
| **API changes in SEA-RAFT** | Low | Medium | Pin to specific git commit/version |
| **User confusion (too many models)** | Medium | Low | Clear tooltips, set good default (sea-raft-medium) |
| **PyTorch version conflicts** | Medium | High | Document requirements, test on multiple versions |

**Overall Risk:** **LOW-MEDIUM** - SEA-RAFT API is very similar to RAFT, integration is straightforward

---

## Success Metrics

**Technical:**
- [ ] SEA-RAFT models load without errors
- [ ] Inference speed 2-3x faster than RAFT
- [ ] Output quality visibly better (edge preservation)
- [ ] All existing RAFT workflows still work
- [ ] Model downloads work on first run

**User Experience:**
- [ ] Users can switch models with one dropdown change
- [ ] Clear guidance on which model to choose
- [ ] First-time setup is < 5 minutes
- [ ] Error messages are actionable

**Documentation:**
- [ ] README explains RAFT vs SEA-RAFT clearly
- [ ] Example workflows use SEA-RAFT
- [ ] Troubleshooting guide covers common issues
- [ ] License/attribution is complete

---

## Alternative Approaches Considered

### Option A: Separate Node for SEA-RAFT ❌

**Pros:** Clean separation, no compatibility risks
**Cons:** Code duplication, confusing for users ("which node do I use?")
**Decision:** Rejected - Better to unify in single node

### Option B: Replace RAFT Entirely ❌

**Pros:** Simpler codebase, force best practices
**Cons:** Breaks existing workflows, users may have RAFT checkpoints
**Decision:** Rejected - Support both for backward compatibility

### Option C: Unified Node with Model Selection ✅ CHOSEN

**Pros:** Backward compatible, easy to switch, clear UI
**Cons:** Slightly more complex loading logic
**Decision:** Accepted - Best balance of compatibility and features

---

## Open Questions

1. **Model Versioning:** Should we pin SEA-RAFT to a specific commit or use latest?
   - **Recommendation:** Pin to specific tag/commit for stability

2. **Default Model:** Change default from `raft-sintel` to `sea-raft-medium`?
   - **Recommendation:** Yes, but add migration guide for existing users

3. **Checkpoint Storage:** Use HuggingFace cache or custom directory?
   - **Recommendation:** HuggingFace cache (standard location), but allow override

4. **Uncertainty Output:** Expose as separate output or replace confidence?
   - **Recommendation:** Replace confidence (SEA-RAFT's uncertainty is better)

5. **Iteration Count:** Should UI show different ranges for RAFT vs SEA-RAFT?
   - **Recommendation:** Keep unified range (6-32), auto-adjust default

---

## Next Steps

**Immediate (< 1 week):**
1. ✅ Create this implementation plan
2. Set up test environment with SEA-RAFT
3. Implement Phase 1 (Foundation)
4. Test basic model loading and inference

**Short-term (1-2 weeks):**
5. Implement Phase 2 (Integration)
6. Update existing workflows to test both RAFT and SEA-RAFT
7. Gather performance benchmarks

**Medium-term (2-4 weeks):**
8. Implement Phase 3 (Optimization)
9. Create example workflows with SEA-RAFT
10. Write documentation

**Before Release:**
11. Test on multiple GPUs (8GB, 12GB, 24GB VRAM)
12. Verify installation on fresh Python environment
13. Complete Phase 4 (Documentation & Polish)
14. Git commit and push to repository

---

## Conclusion

**SEA-RAFT integration is HIGHLY RECOMMENDED:**

✅ **Pros:**
- Significant speed improvement (2.3x faster)
- Better accuracy (22% error reduction)
- Easy integration (API nearly identical to RAFT)
- HuggingFace Hub support (easy downloads)
- Modern architecture (ResNet-based)
- Active maintenance (2024 release)

⚠️ **Cons:**
- Requires PyTorch 2.2.0+ (may need upgrade)
- Adds dependency on huggingface-hub
- Slightly more complex model selection UI

**Effort:** 16-23 hours total
**Impact:** High - meaningful improvement for all users
**Risk:** Low-Medium - straightforward integration

**Recommendation:** Proceed with implementation following the 4-phase plan above.

---

## References

- **SEA-RAFT Paper:** https://arxiv.org/abs/2405.14793
- **SEA-RAFT GitHub:** https://github.com/princeton-vl/SEA-RAFT
- **Original RAFT Paper:** https://arxiv.org/abs/2003.12039
- **RAFT GitHub:** https://github.com/princeton-vl/RAFT
- **HuggingFace Models:** https://huggingface.co/MemorySlices

---

*Plan created: 2025-01-XX*
*Status: READY FOR IMPLEMENTATION*
