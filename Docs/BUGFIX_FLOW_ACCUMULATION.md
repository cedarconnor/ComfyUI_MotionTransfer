# Critical Bug Fix: Flow Accumulation for Motion Transfer

**Date:** 2025-10-23
**Version:** v0.6.2
**Severity:** Critical - Affected all motion transfer outputs

## Problem Description

The motion transfer output showed frames that appeared to "reset" to the original still image on each frame, with only minimal displacement applied. This made the motion look jittery and incorrect, as if the still image wasn't properly accumulating motion from the source video.

### Visual Symptoms
- Output frames looked almost identical to the source still
- Motion appeared as small jitters rather than smooth accumulated movement
- Each frame seemed to "reset" instead of building on previous motion
- The effect looked like the still image was being warped independently for each frame

## Root Cause

The bug was in the **FlowToSTMap** node (lines 344-385) and **SequentialMotionTransfer** node (lines 1067-1125).

### The Misunderstanding

**RAFT optical flow outputs:** Frame-to-frame displacement vectors
- `flow[0]` = displacement from frame 0 → frame 1
- `flow[1]` = displacement from frame 1 → frame 2
- `flow[2]` = displacement from frame 2 → frame 3
- etc.

**Motion transfer requires:** Total displacement from the **original still image**
- Frame 0: No displacement (identity)
- Frame 1: `flow[0]` (total displacement)
- Frame 2: `flow[0] + flow[1]` (accumulated displacement)
- Frame 3: `flow[0] + flow[1] + flow[2]` (accumulated displacement)
- etc.

### What The Old Code Did (WRONG)

```python
for i in range(batch_size):
    flow_frame = flow[i]  # [H, W, 2]
    flow_u = flow_frame[:, :, 0]
    flow_v = flow_frame[:, :, 1]

    # BUG: Using frame-to-frame flow directly!
    new_x = x_coords + flow_u
    new_y = y_coords + flow_v
```

This applied each flow field **independently**, so:
- Frame 1: Warped by `flow[0]` (small displacement from original still)
- Frame 2: Warped by `flow[1]` (small displacement from original still) ❌ WRONG!
- Frame 3: Warped by `flow[2]` (small displacement from original still) ❌ WRONG!

Each frame was only displaced by the **current frame's motion**, not the **total accumulated motion** from the beginning.

## The Fix

### Fix #1: FlowToSTMap Node (motion_transfer_nodes.py:344-398)

Added flow accumulation loop:

```python
# Accumulate flow vectors for motion transfer
accumulated_flow_u = np.zeros((height, width), dtype=np.float32)
accumulated_flow_v = np.zeros((height, width), dtype=np.float32)

stmaps = []
for i in range(batch_size):
    # Accumulate current flow onto total displacement
    accumulated_flow_u += flow[i, :, :, 0]
    accumulated_flow_v += flow[i, :, :, 1]

    # Use accumulated flow for STMap
    new_x = x_coords + accumulated_flow_u
    new_y = y_coords + accumulated_flow_v
```

Now:
- Frame 1: Warped by `flow[0]` ✓
- Frame 2: Warped by `flow[0] + flow[1]` ✓
- Frame 3: Warped by `flow[0] + flow[1] + flow[2]` ✓

### Fix #2: SequentialMotionTransfer Node (motion_transfer_nodes.py:1065-1133)

The sequential processing node needed a similar fix, but it processes one frame at a time, so accumulation is done manually:

```python
# Initialize accumulation buffers
accumulated_flow_u = None
accumulated_flow_v = None

for t in range(batch_size - 1):
    # ... RAFT flow extraction ...

    # Initialize on first frame
    if accumulated_flow_u is None:
        accumulated_flow_u = np.zeros((target_height, target_width), dtype=np.float32)
        accumulated_flow_v = np.zeros((target_height, target_width), dtype=np.float32)

    # Accumulate frame-to-frame flow
    accumulated_flow_u += refined_flow[:, :, 0]
    accumulated_flow_v += refined_flow[:, :, 1]

    # Build STMap directly from accumulated flow
    new_x = x_coords + accumulated_flow_u
    new_y = y_coords + accumulated_flow_v
```

## Impact

This bug affected **all motion transfer workflows** using:
- Pipeline A (Flow-Warp): FlowToSTMap node
- SequentialMotionTransfer: All-in-one processing

The bug did NOT affect:
- Pipeline B (Mesh-Warp): Uses mesh deformation, not direct flow
- Pipeline B2 (CoTracker): Uses point tracking, not optical flow
- Pipeline C (3D-Proxy): Uses depth reprojection (but has similar conceptual issue)

## Expected Behavior After Fix

After this fix, motion should accumulate properly:
1. Frame 1 shows small displacement from original still
2. Frame 2 shows accumulated displacement (more motion)
3. Frame 3 shows even more accumulated displacement
4. Motion builds smoothly throughout the sequence

The output should now look like the still image is **actually animated** by the source video motion, rather than jittering in place.

## Testing Recommendations

To verify the fix works:

1. **Visual Test:** Run your existing workflow again
   - Output frames should show increasing displacement from the original still
   - Motion should accumulate smoothly across frames
   - No more "resetting" to the original image

2. **Diagnostic Test:** Check STMap values
   - Frame 1 STMap should have values close to identity (0.5, 0.5 in center)
   - Frame 2 STMap should have larger deviations
   - Frame N STMap should have largest deviations
   - Values should gradually diverge, not reset

3. **Comparison Test:**
   - Old output: Frames looked almost identical, slight jitter
   - New output: Frames should show clear accumulated motion

## Files Modified

1. **motion_transfer_nodes.py:344-398** - FlowToSTMap.to_stmap()
   - Added flow accumulation loop
   - Updated docstring to explain accumulation

2. **motion_transfer_nodes.py:1065-1133** - SequentialMotionTransfer.run()
   - Added accumulated_flow_u/v buffers
   - Accumulate flow on each frame
   - Build STMap directly from accumulated flow

3. **README.md:310-316** - FlowToSTMap documentation
   - Added explanation of automatic flow accumulation
   - Clarified the difference between RAFT output and motion transfer needs

4. **BUGFIX_FLOW_ACCUMULATION.md** (this file)
   - Complete documentation of the bug and fix

## Related Concepts

This is similar to the difference between:
- **Velocity** (frame-to-frame) vs **Position** (accumulated from origin)
- **Delta encoding** vs **Absolute encoding**
- **Differential backup** vs **Full backup**

RAFT gives you velocity (how much things moved *this frame*), but motion transfer needs position (where things are *relative to the start*).

## Version History

- **v0.6.1 and earlier:** Bug present - incorrect frame-to-frame flow usage
- **v0.6.2:** Bug fixed - proper flow accumulation for motion transfer

## Credits

Bug discovered and fixed on 2025-10-23 based on user report: "the output looks wrong, it looks like the source frame is resetting on each frame."
