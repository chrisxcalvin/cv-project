# Calibration data

Real photos captured by the team to replace the guessed thresholds in
`utils/constants.py` with numbers actually fit to real real/print/replay
samples, instead of the LFW + synthetic-degradation numbers the toy CNN
checkpoint currently uses.

**These images are NOT committed to git - this repo is public, and
photos of our faces should not go on it.** Everyone captures locally,
then shares their photos with the team over a private channel (shared
drive folder, not git). Only the folder structure (`.gitkeep`), this
README, and the calibration scripts/output are tracked in git.

## Folder = category

```
calibration_data/
  real/     genuine face, live in front of the webcam
  print/    a printed photo of a face, held up to the webcam
  replay/   a face shown on a second screen (phone/laptop), held up to the webcam
```

## Naming convention

`<yourname>_<category>_<n>.jpg`, e.g.:

```
real/vansh_real_1.jpg
real/vansh_real_2.jpg
print/vansh_print_1.jpg
replay/vansh_replay_1.jpg
```

The name prefix isn't read by anything - it's just so we can tell whose
photos are whose while reviewing the results.

## How to capture

Use `scripts/capture_calibration_photo.py` (opens the SAME webcam the app
uses) rather than a phone camera roll - see the top-level explanation for
why. Run from the project root:

```
.venv\Scripts\python.exe scripts\capture_calibration_photo.py --name vansh --category real
.venv\Scripts\python.exe scripts\capture_calibration_photo.py --name vansh --category print
.venv\Scripts\python.exe scripts\capture_calibration_photo.py --name vansh --category replay
```

A preview window opens - press SPACE to save a frame (auto-numbered),
press Q to quit. Aim for 5-6 saved frames per category per person.

## Sharing your photos with the team

Do NOT `git add`/commit these images. Instead:

1. Zip your local `calibration_data/` folder.
2. Upload the zip to the team's shared private drive folder (Drive/OneDrive -
   whatever the team already uses).
3. Whoever is running the analysis (Vansh) downloads everyone's zip and
   merges them into their own local `calibration_data/real|print|replay/`
   folders. The `<name>_<category>_<n>.jpg` naming convention keeps
   everyone's files from colliding.

## What happens next

Once the folders are populated, run:

```
.venv\Scripts\python.exe scripts\analyze_calibration_data.py
```

This prints the real/print/replay feature distributions and suggested
threshold values for `utils/constants.py`.
