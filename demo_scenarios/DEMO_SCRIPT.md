# DeepCheck Demo Day Script

Run the dashboard first: `python main.py` (or `python main.py --live` for the raw
OpenCV window with overlays). Walk through the four scenarios below in order.

## Scenario 1 - Real face (expect: ACCEPT)

1. Sit at a normal distance from the webcam, face clearly lit and visible.
2. Capture a photo in the Live Demo tab.
3. Expected result: `real_confidence` well above 0.75 - rich ORB keypoints,
   healthy texture variance, low FFT high-frequency ratio, dense Canny edges.
4. Verdict card shows **ACCEPT** immediately, no active challenge triggered.

## Scenario 2 - Print attack (expect: REJECT)

1. Print a clear photo of a face (or use a printed photo you already have).
2. Hold it flat in front of the webcam, filling roughly the same frame area
   as a real face would.
3. Capture a photo.
4. Expected result: low texture variance (Otsu - flat paper regions), fewer
   ORB keypoints, elevated FFT high-frequency ratio (halftone dot pattern).
5. Verdict card shows **REJECT**.

## Scenario 3 - Replay attack (expect: REJECT)

1. Display a face photo full-screen on a phone or tablet.
2. Hold the screen in front of the webcam at a similar distance/framing.
3. Capture a photo.
4. Expected result: strong FFT high-frequency energy from the pixel/subpixel
   grid and moire patterns, reduced or grid-like ORB keypoints.
5. Verdict card shows **REJECT**.

## Scenario 4 - Borderline case (expect: challenge -> then REJECT)

1. Use a high-quality printed photo or a photo held at an angle/lighting that
   pushes the passive score into the uncertain band (0.30 - 0.75). Slight
   glare, mid-distance framing, or a glossy print tend to land here.
2. Capture the initial photo - dashboard shows "Challenge Required" with a
   randomly chosen prompt (blink / turn left / turn right / open mouth).
3. Since it's a static photo, it cannot blink, turn, or open its mouth -
   capture the challenge photo without complying with the prompt.
4. Expected result: active challenge fails (`passed: false`), final verdict
   is **REJECT** even though the passive score alone was ambiguous.

## Scenario 5 - Multi-face guard (expect: rejected outright, no scoring)

1. Hold a second face (a printed photo, or another person) in frame alongside
   your own face.
2. Capture a photo.
3. Expected result: the dashboard reports "N faces detected" and stops before
   running the passive/active pipeline at all - this blocks the common trick
   of keeping a spoof photo ready next to your real face.

## Tips for a smooth demo

- Good, even lighting makes Scenario 1 land solidly in ACCEPT range.
- If Scenario 2/3 come back ACCEPT, move the print/screen closer so it fills
  more of the frame (spoof artifacts are easier to pick up at close range).
- On a REJECT/uncertain result, point out the "Likely spoof type" caption
  (printed_photo vs. screen_replay) and the explainability bar chart - both
  are heuristic sub-classifications, good talking points for "why" questions.
- Download the verdict report (PNG button under Final Verdict) as a leave-
  behind artifact for whoever's grading.
- Check the Analytics tab afterward to show the running accept/reject/
  challenge-trigger rates, and now spoof_type breakdown, across all scenarios.
