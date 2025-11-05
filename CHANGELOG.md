# Changelog

All notable changes to the ComfyUI Motion Transfer project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive unit test suite with pytest
  - Tests for flow accumulation logic (FlowToSTMap)
  - Tests for tiled warping and feather blending (TileWarp16K)
  - Tests for flow upsampling (FlowSRRefine)
  - Tests for mesh generation (MeshBuilder2D)
  - Test fixtures and configuration
- Development dependencies (`requirements-dev.txt`)
- Pytest configuration (`pytest.ini`)

### Changed
- Repository structure improvements in progress
- Code quality enhancements planned

---

## [0.7.0] - 2025-10-23

### Added
- **CUDA acceleration** for 5-10× performance improvement
  - GPU-accelerated TileWarp16K kernel (8-15× faster)
  - GPU-accelerated BarycentricWarp kernel (10-20× faster)
  - GPU-accelerated FlowSRRefine kernel (3-5× faster)
  - Total pipeline speedup: 40 min → 5-7 min for typical 16K workflows
- CUDA installation scripts (`build.bat` for Windows, `build.sh` for Linux/macOS)
- Comprehensive CUDA documentation in `cuda/README.md`
- Automatic fallback to CPU implementation if CUDA unavailable

### Fixed
- CUDA compilation issues and kernel optimization

### Performance
- **Major speedup**: 16K motion transfer pipeline now runs in ~5-7 minutes vs ~40 minutes (8× faster)
- Reduced memory usage with optimized CUDA kernels

---

## [0.6.1] - 2025-10-21

### Fixed
- Critical bug fixes in model loading
- Cleanup and stability improvements
- Edge case handling in flow processing
- Memory management optimizations

### Changed
- Updated documentation to reflect architecture improvements
- Improved error messages and debugging output

---

## [0.6.0] - 2025-10-21 - **Architecture Refactor Release**

### Added
- **Modular model loading system**
  - New `models/` package with clean architecture
  - `models/raft_loader.py` - Unified RAFT model loader
  - `models/searaft_loader.py` - SEA-RAFT HuggingFace integration
  - `models/__init__.py` - OpticalFlowModel abstraction
- **Full dual-model support**
  - Both RAFT and SEA-RAFT working seamlessly
  - Automatic model type detection
  - Model caching for performance
- GitHub Actions workflows
  - Claude Code Review workflow
  - Claude PR Assistant workflow

### Changed
- **Complete RAFT/SEA-RAFT architecture refactor**
  - Replaced 210 lines of complex path detection with 12 lines using model loaders
  - Removed sys.path manipulation (now uses proper Python imports)
  - 94% reduction in model loading code complexity
  - Bundled vendor code now actually used (no external repos needed)
- Updated `README.md` with v0.6.0 architecture details
- Updated `Docs/agents.md` with new architecture overview

### Fixed
- SEA-RAFT now fully functional (was listed but broken in v0.5)
- Model loading reliability improved
- Cleaner error messages with actionable guidance

### Performance
- SEA-RAFT: 2.3x faster than RAFT (6 min vs 14 min for 120 frames)
- Better model caching reduces reload time

---

## [0.5.0] - 2025-10-20

### Added
- **Pipeline B2: CoTracker Mesh-Warp** (ECCV 2024)
  - `MeshFromCoTracker` node - Converts point trajectories to mesh
  - Integration with Meta AI's CoTracker3 transformer-based tracking
  - Superior temporal stability for organic motion (faces, hands, cloth)
  - Handles occlusions better than optical flow
  - Example workflow: `examples/workflow_pipeline_b2_cotracker.json`
  - Comprehensive documentation: `Docs/COTRACKER_GUIDE.md` (645 lines)
  - Integration plan: `Docs/COTRACKER_INTEGRATION_PLAN.md`
- SEA-RAFT support (ECCV 2024)
  - 2.3× faster optical flow extraction vs original RAFT
  - 22% more accurate (ECCV 2024 Best Paper Award Candidate)
  - HuggingFace Hub auto-download (no manual model download)
  - Better edge preservation for high-res warping
  - Three model sizes: small (8GB), medium (12-24GB), large (24GB+)
- Documentation improvements
  - `Docs/SEA_RAFT_INTEGRATION_PLAN.md`
  - `Docs/CoTracker_MeshWarp_DesignDoc.md`
  - Updated README with Pipeline B2 section

### Fixed
- PIL decompression bomb protection disabled for ultra-high-res images
  - Allows loading 16K+ images (>178 megapixels)
  - Safe for user-provided inputs in controlled environment
- RAFT auto-detection and installation issues
  - Intelligent path searching for RAFT location
  - Clear installation instructions in error messages
- requirements.txt dependency conflicts
  - Removed strict version pins to avoid conflicts with other custom nodes
  - Made OpenEXR optional (can fail on some Windows systems)
  - Works with wide range of numpy/opencv versions
- Critical bugs and edge cases
  - Flow indexing errors
  - Mesh degenerate triangle filtering
  - Directory case mismatches

### Changed
- Added MIT LICENSE file
- Simplified dependency management
- Improved installation documentation

---

## [0.4.0] - 2025-10-20

### Added
- Complete ComfyUI integration
  - All 12 nodes registered with proper categories
  - Node tooltips with detailed parameter descriptions
  - Example workflows for all pipelines
- Comprehensive documentation
  - `README.md` - 613 lines with installation, usage, troubleshooting
  - `Docs/IMPLEMENTATION_PLAN.md` - Technical summary
  - `Docs/TOOLTIPS.md` - Complete parameter reference
  - `examples/README.md` - Workflow usage guide

### Changed
- Improved node parameter descriptions
- Better error handling and validation
- Enhanced user experience

---

## [0.3.0] - 2025-10-20

### Added
- **Pipeline C: 3D-Proxy** (Experimental)
  - `DepthEstimator` node - Monocular depth estimation (placeholder)
  - `ProxyReprojector` node - Depth-based reprojection for parallax
  - Example workflow: `examples/workflow_pipeline_c_proxy.json`
- Framework for depth model integration (MiDaS/DPT)
- Parallax handling for camera motion

### Changed
- Improved depth-based warping algorithm
- Better handling of foreground/background separation

---

## [0.2.0] - 2025-10-20

### Added
- **Pipeline B: Mesh-Warp** (Advanced)
  - `MeshBuilder2D` node - Delaunay triangulation from flow
  - `AdaptiveTessellate` node - Flow gradient-based mesh refinement
  - `BarycentricWarp` node - Triangle-based warping
  - Example workflow: `examples/workflow_pipeline_b_mesh.json`
- Better handling of large deformations
- More stable warping at edges compared to pixel-based flow

### Performance
- Mesh-based warping better for character animation and fabric/cloth
- Slower than Pipeline A but more stable for extreme motion

---

## [0.1.0] - 2025-10-20 - **Initial Release**

### Added
- **Pipeline A: Flow-Warp** (Production Ready)
  - `RAFTFlowExtractor` node - Dense optical flow using RAFT
  - `FlowSRRefine` node - Edge-aware flow upsampling with guided filtering
  - `FlowToSTMap` node - Flow to STMap conversion with accumulation
  - `TileWarp16K` node - Tiled warping with seamless feathered blending
  - `TemporalConsistency` node - Flow-based temporal stabilization
  - `HiResWriter` node - Multi-format export (PNG/EXR/JPG)
- Support for ultra-high-resolution images (16K+, up to 32K)
- Tiled processing with 2048×2048 tiles and 128px feathered overlap
- Example workflow: `examples/workflow_pipeline_a_flow.json`

### Features
- **Three complementary pipelines**:
  - Pipeline A (Flow-Warp): Fast, general-purpose, production-ready
  - Pipeline B (Mesh-Warp): Advanced, for large deformations
  - Pipeline C (3D-Proxy): Experimental, for parallax/camera motion
- **RAFT optical flow integration**
  - Bundled RAFT vendor code (`raft_vendor/`)
  - Manual model weight download from Dropbox
  - Configurable iteration count (12-20)
- **High-resolution support**
  - Tested up to 16K (15360×8640)
  - Supports 4K, 8K, 16K, and custom resolutions
  - Memory-efficient tiled processing
- **Quality features**
  - Guided filtering for edge-aware flow upsampling
  - Temporal consistency to reduce flicker
  - Confidence maps for flow reliability
  - Multiple interpolation modes (linear, cubic, lanczos4)
- **Export formats**
  - PNG (8-bit, sRGB)
  - EXR (16-bit half, linear) - optional
  - JPG (8-bit, quality 95)
- **Documentation**
  - Comprehensive README with installation guide
  - Troubleshooting section
  - Performance optimization tips
  - Design documents in `Docs/` directory

### Known Limitations
- Pipeline C uses placeholder depth estimation (real depth models planned)
- CUDA acceleration not yet implemented (planned for v0.7)
- No progress indicators yet (planned for v0.8)

---

## Version History Summary

- **v0.7.0** (2025-10-23): CUDA acceleration - 5-10× speedup
- **v0.6.1** (2025-10-21): Bug fixes and stability
- **v0.6.0** (2025-10-21): Architecture refactor - modular model loading
- **v0.5.0** (2025-10-20): Pipeline B2 (CoTracker) + SEA-RAFT integration
- **v0.4.0** (2025-10-20): ComfyUI integration complete
- **v0.3.0** (2025-10-20): Pipeline C (3D-Proxy) experimental
- **v0.2.0** (2025-10-20): Pipeline B (Mesh-Warp) advanced
- **v0.1.0** (2025-10-20): Initial release - Pipeline A production-ready

---

## Roadmap

### Planned for v0.8
- [ ] Progress indicators and status updates
- [ ] Real-time ETA calculations
- [ ] Memory usage monitoring
- [ ] Interactive parameter tuning with preview mode

### Planned for v1.0 (Production Release)
- [ ] Real depth model integration (MiDaS/DPT)
- [ ] ProRes/DNxHR video encoding
- [ ] Alembic mesh export for 3D software
- [ ] ACES/OCIO color management
- [ ] Comprehensive testing and validation
- [ ] Community feedback integration

### Future Enhancements
- [ ] Forward-backward flow consistency checking
- [ ] Occlusion detection and temporal inpainting
- [ ] Global affine stabilization
- [ ] Multi-GPU support
- [ ] TorchScript model compilation
- [ ] Docker containerization
- [ ] Web UI components

---

## Credits

### Algorithms & Research
- **SEA-RAFT**: Wang, Lipson, Deng (Princeton, ECCV 2024) - https://github.com/princeton-vl/SEA-RAFT
- **RAFT**: Teed & Deng (ECCV 2020) - https://github.com/princeton-vl/RAFT
- **CoTracker**: Meta AI (Karaev, Rocco, et al., ECCV 2024) - https://github.com/facebookresearch/co-tracker
- **Guided Filter**: He et al. (ECCV 2015)

### Implementation
- **Architecture inspiration**: alanhzh/ComfyUI-RAFT for clean relative imports
- **ComfyUI integration**: Custom node development framework
- **Development**: AI-assisted implementation (Claude Code)

---

[Unreleased]: https://github.com/cedarconnor/ComfyUI_MotionTransfer/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/cedarconnor/ComfyUI_MotionTransfer/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/cedarconnor/ComfyUI_MotionTransfer/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/cedarconnor/ComfyUI_MotionTransfer/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/cedarconnor/ComfyUI_MotionTransfer/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/cedarconnor/ComfyUI_MotionTransfer/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/cedarconnor/ComfyUI_MotionTransfer/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/cedarconnor/ComfyUI_MotionTransfer/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cedarconnor/ComfyUI_MotionTransfer/releases/tag/v0.1.0
