# Bench capture (capture-to-train data)

Records one human task from three cameras at once, lines the streams up, and
checks each take before you rely on it as training data.

| Stream | Device | How it starts |
|---|---|---|
| depth + color | Intel RealSense D455 on the recording host (USB 3) | `python -m capture take` records it |
| first-person view | phone on your chest, 1080p30 wide | you press record |
| wide third-person view | MacBook webcam (QuickTime) | you press record |

The phone and MacBook are started by hand and send their clips to the machine running ingest. For example, use **Taildrop**
when the devices share a tailnet.

## A session

```bash
cd ~/Desktop/Bench-Capture               # or wherever this repo is checked out
source ~/.venvs/capture/bin/activate     # optional: adds mediapipe for the hands check

python -m capture devices                # RealSense found, and on USB 3?

python -m capture take bench-2026-11-29 --task "sort 5 bolts into 2 trays"
#  1. start the phone + MacBook recordings, press Enter
#  2. light OFF for about a second, light back ON, clap once, do the task
#  3. press Enter to stop; stop the phone + MacBook
#  4. Transfer both clips to the machine running ingest
# ...repeat `take` for each rep (5 unedited reps)

python -m capture ingest bench-2026-11-29
python -m capture validate bench-2026-11-29
```

Sessions live in `~/captures/<session>/` (override with `--root` or `$CAPTURE_ROOT`).
They are data, not code, and are not committed.

## Lighting and the sync signal

Set up the lights first:

```bash
python -m capture light-check --seconds 30
```

It prints the RealSense brightness twice a second, marks `<< FLICK` whenever a light
change would pass the sync check, and ends with WOULD PASS / would FAIL. Nothing is
recorded. Aim the camera at what the light actually hits.

The D455 has no microphone, so a clap cannot line it up with the other two. The
**light flick** can, in either direction:

* **lamp off for ~1 s, then back on** (daytime or a lit room), or
* **flashlight on for ~1 s, then off** (dark room).

The first such edge in each clip is the sync point, so every camera locks to the same
physical event; if two cameras lock to different directions the take fails
(`sync_same_edge`). An edge counts when brightness changes by at least 20% of the
brighter level and at least 6 levels (out of 255), held about a quarter second, so
a small lamp in a dark room works while a single auto-exposure frame, a slow fade, or
someone walking past does not. The flick must happen in the first minute of every clip.

Each stream's brightness is recorded in the manifest. Below 40/255 median the take
gets a lighting WARN (depth still works in the dark, the D455 projects its own IR
pattern, but color and hand detection get poor); above 225 it is blown out.

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
| `sync_same_edge` | streams locked to flicks of different direction (FAIL) |
| `*_lighting` | median brightness under 40 or over 225 (WARN) |
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
