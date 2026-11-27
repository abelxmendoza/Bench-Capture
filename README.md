# Bench Capture

Bench Capture records and validates human demonstrations from a RealSense D455
and two independent color cameras. The recorder saves color and depth locally,
aligns the streams using a shared light change, and writes a take manifest with
timing, hashes, and validation results. It can run without Omega-Code.

## Quick start

Use Python 3 with NumPy and FFmpeg/FFprobe. The recording host also needs a
working `pyrealsense2` installation for the connected D455. MediaPipe is
optional for the hands-in-frame check. Close other programs using the D455
before starting an SDK capture.

```bash
python3 -m capture devices
python3 -m capture light-check --seconds 30
python3 -m capture take bench-1 --task "sort five bolts into two trays"
```

Start the phone and MacBook views by hand. Once all three are recording,
make the same visible light change in every view, then perform the task.
See the [operator guide](capture/README.md) for lighting, audio cross-checks,
and ingest. By default sessions live under `~/captures`. Move the external
clips to the machine running ingest, then run:

```bash
python3 -m capture ingest bench-1
python3 -m capture validate bench-1
```

The default workflow keeps media local. Hivemind logging is an optional,
explicit step. The extracted software has automated tests. A complete
three-camera hardware take remains to be validated.

## Development

```bash
python3 -m pytest -q tests/capture
```

This repo is the place to develop and test demonstration synchronization,
multi-camera calibration, triangulation, and trajectory extraction. The
workspace scan and the simulator have independent inputs and outputs.
Future trajectory estimates need calibrated camera transforms, a workspace
reference frame, and an independent accuracy check.

The initial `capture/` implementation and tests were extracted with their
relevant Git history from [Omega-Code at 904d00a](https://github.com/abelxmendoza/Omega-Code/commit/904d00a7207d747d06fd8965da124c4caffd278c).
