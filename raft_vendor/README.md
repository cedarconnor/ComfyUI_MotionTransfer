# Vendored RAFT Code

This directory contains code from the RAFT (Recurrent All-Pairs Field Transforms) optical flow project.

**Source:** https://github.com/princeton-vl/RAFT
**License:** BSD-3-Clause (see LICENSE file)
**Authors:** Zachary Teed and Jia Deng (Princeton University)
**Paper:** ECCV 2020

## Citation

```bibtex
@inproceedings{teed2020raft,
  title={RAFT: Recurrent All-Pairs Field Transforms for Optical Flow},
  author={Teed, Zachary and Deng, Jia},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

## Why Vendored?

This code is included directly in the Motion Transfer package to:
1. Simplify installation (no manual git clone required)
2. Ensure version compatibility
3. Reduce setup steps for users

The code is unmodified from the original RAFT repository.
