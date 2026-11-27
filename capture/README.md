# Bench capture (capture-to-train data)

Records one human task from three cameras at once, lines the streams up, and
checks each take before you rely on it as training data.

| Stream | Device | How it starts |
|---|---|---|
| depth + color | Intel RealSense D455 on scythe (USB 3) | `python -m capture take` records it |
| first-person view | phone on your chest, 1080p30 wide | you press record |
| wide third-person view | MacBook webcam (QuickTime) | you press record |

The phone and MacBook are started by hand and send their clips to scythe with
**Taildrop** (Share → Taildrop → `scythe-1`), so nothing needs SSH access into the Mac.

## A session

```bash
cd ~/Desktop/Omega-Code-capture          # or wherever this branch is checked out
source ~/.venvs/capture/bin/activate     # optional: adds mediapipe for the hands check

python -m capture devices                # RealSense found, and on USB 3?

python -m capture take bench-2026-11-29 --task "sort 5 bolts into 2 trays"
#  1. start the phone + MacBook recordings, press Enter
#  2. light OFF for about a second, light back ON, clap once, do the task
#  3. press Enter to stop; stop the phone + MacBook
#  4. Taildrop both clips to scythe-1
# ...repeat `take` for each rep (5 unedited reps)

python -m capture run bench-2026-11-29 --push   # ingest + validate + log to hivemind
```

Sessions live in `~/captures/<session>/` (override with `--root` or `$CAPTURE_ROOT`).
They are data, not code, and are not committed.

## The sync signal

The D455 has no microphone, so a clap cannot line it up with the other two. The
**light flick** can: switching the room light off drops every camera's brightness
within a frame or two. Keep the light off for about a second; a blink shorter than
~0.25 s is ignored on purpose, so a single dark frame from auto-exposure is never
mistaken for it. The flick must happen in the first minute of every clip.

The clap is a second, independent check between the phone and MacBook audio: the gap
from flick to clap must agree within 100 ms in both, or the take gets a warning.

## What `validate` checks

Each take gets `take_NN/manifest.json` (file hashes, frame rates, sync offsets, every
check) and the session gets `summary.md`.

| check | FAIL / WARN when |
|---|---|
| `files_present` | a device's file is missing (FAIL) |
| `depth_present` | no depth frames, or under 30% valid depth pixels (FAIL) |
| `realsense_*_drops` | more than 1% of frames dropped (FAIL) |
| `phone_frame_rate`, `mac_frame_rate` | effective fps under 90% of nominal (FAIL) |
| `sync_light_flick` | the flick is not found in some stream (FAIL) |
| `aligned_overlap` | stop times differ by more than 2 s (WARN) |
| `clap_crosscheck` | phone and Mac disagree by more than 100 ms, or no clap heard (WARN) |
| `hands_in_frame` | hands found in under 60% of sampled phone frames (FAIL; SKIP without mediapipe) |

A take with any FAIL should be redone. `t = 0` in the aligned data is the light-off
edge; subtract a stream's `sync_offset_s` from its own timestamps.

## Hivemind

`log` (and `run`) writes one row per session to
`<agent-hivemind>/state/capture_sessions.csv`; `--push` commits and pushes that file.
The clone is found at `$HIVEMIND_DIR`, `~/agent-hivemind` or `~/Desktop/agent-hivemind`.

## Tests

`python3 -m pytest tests/capture` - synthetic clips built with ffmpeg; no camera needed.
