#!/usr/bin/env python3
"""Run a trained RoadWatch detector over real footage and report what it saw.

The test split comes from the same distribution as training, so its mAP says very
little about New Jersey roads shot through a phone mount. This runs the model over
an actual clip and writes the three things that decide whether it is usable: an
annotated video to eyeball, a per-detection CSV, and a box-height breakdown.

The height breakdown says how small the boxes it did find are, not what it missed:
every height in it is conditional on the detection succeeding. To find out what
imgsz=960 actually buys over 640, run the same clip twice and diff the CSVs.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# YOLO's finest detection head has stride 8, so an object smaller than a couple of
# cells on the model's input grid has almost no chance. Box heights are reported in
# model-input pixels against this so the far-field question has a concrete answer.
STRIDE_FLOOR = 8


def frames_of(capture, every):
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            return
        if index % every == 0:
            yield index, frame
        index += 1


def histogram(values, edges, width=40):
    counts = Counter()
    for value in values:
        slot = int(np.searchsorted(edges, value, side="right")) - 1
        counts[min(max(slot, 0), len(edges) - 2)] += 1
    peak = max(counts.values()) if counts else 1
    lines = []
    for i in range(len(edges) - 1):
        n = counts[i]
        bar = "#" * int(round(width * n / peak)) if n else ""
        lines.append(f"  {edges[i]:5.2f} to {edges[i + 1]:5.2f}  {bar:<{width}} {n}")
    return "\n".join(lines)


def report(rows, names, processed, seconds, scale):
    out = [f"frames scored: {processed}, covering {seconds:.1f}s of footage"]
    if not rows:
        out.append("\nno detections at this threshold. either the model does not "
                   "transfer to this footage or --conf is too high.")
        return "\n".join(out)

    per_class = Counter(r["cls_name"] for r in rows)
    for name in names:
        if not per_class[name]:
            continue
        rate = per_class[name] / seconds * 60 if seconds else 0
        out.append(f"{name}: {per_class[name]} detections, {rate:.1f} per minute of footage")
    silent = [n for n in names if not per_class[n]]
    if silent:
        out.append(f"never detected: {', '.join(silent)}")

    potholes = [r for r in rows if r["cls_name"] == "pothole"]
    if not potholes:
        return "\n".join(out)

    confs = [r["conf"] for r in potholes]
    out.append("\npothole confidence")
    out.append(histogram(confs, np.linspace(min(confs), 1.0, 9)))

    # Height in model-input pixels, not source pixels: that is the space the detector
    # actually works in, and it is what makes the 960-vs-640 tradeoff legible.
    heights = np.array([r["h_px"] for r in potholes]) * scale
    percentiles = np.percentile(heights, [10, 50, 90])
    tiny = int((heights < STRIDE_FLOOR * 2).sum())
    out.append(f"\npothole box height in model-input px (source px x {scale:.3f})")
    out.append(f"  p10 {percentiles[0]:.1f}   p50 {percentiles[1]:.1f}   p90 {percentiles[2]:.1f}")
    out.append(f"  under {STRIDE_FLOOR * 2}px: {tiny} of {len(heights)} "
               f"({tiny / len(heights) * 100:.0f}%), the far-field detections that "
               "dropping to imgsz=640 would cost you")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True, help="best.pt from the Kaggle run")
    parser.add_argument("--source", type=Path, required=True, help="dashcam video file")
    parser.add_argument("--out", type=Path, default=Path("eval_out"))
    parser.add_argument("--conf", type=float, default=0.25,
                        help="detection threshold; sweep this against the notebook's table")
    parser.add_argument("--imgsz", type=int, default=960,
                        help="must match training, or the small far-away boxes vanish")
    parser.add_argument("--device", default=None, help="mps, cpu, or 0; default lets torch pick")
    parser.add_argument("--every", type=int, default=1, help="score every Nth frame")
    parser.add_argument("--max-frames", type=int, default=None, help="stop after N scored frames")
    parser.add_argument("--no-video", action="store_true", help="skip writing the annotated mp4")
    parser.add_argument("--crops", type=Path, default=None,
                        help="dump every detection as a cropped image under DIR/<class>/, "
                             "for eyeballing what a model actually fires on")
    parser.add_argument("--save-negatives", type=Path, default=None,
                        help="write every frame that fired, plus a review crop per detection "
                             "to sort into review/background or review/manhole")
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise SystemExit(f"could not open {args.source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = YOLO(str(args.weights))
    names = [model.names[i] for i in sorted(model.names)]

    args.out.mkdir(parents=True, exist_ok=True)
    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(str(args.out / "annotated.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps / args.every, (width, height))

    # A frame the model fired on wrongly is the most valuable negative available: it is
    # a confuser the model has already proven it cannot handle. Review is sorting, not box
    # drawing: each crop goes into review/background or review/manhole, and the model's own
    # box becomes the manhole label. An empty label would be wrong for a cover, since it
    # teaches the manhole class that covers are background. Anything left unsorted drops
    # its whole frame, because unlabelled road damage trains the model to suppress it.
    negatives = manifest = None
    if args.save_negatives:
        negatives = {name: args.save_negatives / name for name in ("images", "labels", "review")}
        for directory in (*negatives.values(), negatives["review"] / "background",
                          negatives["review"] / "manhole"):
            directory.mkdir(parents=True, exist_ok=True)
        manifest = []

    if args.crops:
        for name in names:
            (args.crops / name).mkdir(parents=True, exist_ok=True)

    rows = []
    processed = 0
    for index, frame in frames_of(capture, args.every):
        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                               device=args.device, verbose=False)[0]
        processed += 1
        fired = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            fired.append((names[cls], float(box.conf[0]), (x1, y1, x2, y2)))
            rows.append({
                "frame": index,
                "time_s": round(index / fps, 3),
                "cls_name": names[cls],
                "conf": round(float(box.conf[0]), 4),
                "w_px": round(x2 - x1, 1),
                "h_px": round(y2 - y1, 1),
                "area_frac": round((x2 - x1) * (y2 - y1) / (width * height), 6),
                # Both fractions are here so a row can be located in the source frame
                # later without re-running the model over it.
                "x_center_frac": round((x1 + x2) / 2 / width, 4),
                # Distance proxy for the severity heuristic: a pothole near the bottom
                # of the frame is close to the car, so its box size means something
                # different than the same box size up near the horizon.
                "y_bottom_frac": round(y2 / height, 4),
            })
        if args.crops and fired:
            for i, (cls_name, conf, (x1, y1, x2, y2)) in enumerate(fired):
                pad_x, pad_y = int((x2 - x1) * 0.6) + 25, int((y2 - y1) * 0.7) + 25
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                ox, oy = max(0, cx - pad_x), max(0, cy - pad_y)
                crop = frame[oy:min(height, cy + pad_y), ox:min(width, cx + pad_x)].copy()
                cv2.rectangle(crop, (int(x1) - ox, int(y1) - oy),
                              (int(x2) - ox, int(y2) - oy), (0, 0, 255), 2)
                cv2.imwrite(str(args.crops / cls_name /
                                f"{conf:.2f}_frame{index:06d}_{i}.jpg"), crop)

        if negatives is not None and fired:
            stem = f"frame_{index:06d}"
            cv2.imwrite(str(negatives["images"] / f"{stem}.jpg"), frame)
            (negatives["labels"] / f"{stem}.txt").write_text("")
            for i, (cls_name, conf, (x1, y1, x2, y2)) in enumerate(fired):
                pad_x, pad_y = int((x2 - x1) * 0.8) + 30, int((y2 - y1) * 0.9) + 30
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                ox, oy = max(0, cx - pad_x), max(0, cy - pad_y)
                crop = frame[oy:min(height, cy + pad_y), ox:min(width, cx + pad_x)].copy()
                cv2.rectangle(crop, (int(x1) - ox, int(y1) - oy),
                              (int(x2) - ox, int(y2) - oy), (0, 0, 255), 2)
                crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                crop_name = f"{stem}__{i}_{cls_name}_{conf:.2f}.jpg"
                cv2.imwrite(str(negatives["review"] / crop_name), crop)
                manifest.append({"frame": index, "det": i, "cls_name": cls_name,
                                 "conf": round(conf, 4), "crop": crop_name,
                                 "x_center": round((x1 + x2) / 2 / width, 6),
                                 "y_center": round((y1 + y2) / 2 / height, 6),
                                 "w": round((x2 - x1) / width, 6),
                                 "h": round((y2 - y1) / height, 6)})

        if writer is not None:
            writer.write(result.plot())
        if processed % 100 == 0:
            print(f"{processed} frames, {len(rows)} detections")
        if args.max_frames and processed >= args.max_frames:
            break

    capture.release()
    if writer is not None:
        writer.release()

    with open(args.out / "detections.csv", "w", newline="") as handle:
        fields = ["frame", "time_s", "cls_name", "conf", "w_px", "h_px", "area_frac",
                  "x_center_frac", "y_bottom_frac"]
        csv.DictWriter(handle, fieldnames=fields).writeheader()
        csv.DictWriter(handle, fieldnames=fields).writerows(rows)

    if manifest is not None:
        with open(args.save_negatives / "manifest.csv", "w", newline="") as handle:
            fields = ["frame", "det", "cls_name", "conf", "crop", "x_center", "y_center", "w", "h"]
            csv.DictWriter(handle, fieldnames=fields).writeheader()
            csv.DictWriter(handle, fieldnames=fields).writerows(manifest)
        print(f"\nnegatives: {len(set(m['frame'] for m in manifest))} frames, "
              f"{len(manifest)} review crops in {args.save_negatives / 'review'}\n"
              "move each crop into review/background or review/manhole, leave road damage "
              "where it is, then pass the directory to prep_dataset.py --negatives")

    scale = args.imgsz / max(width, height)
    summary = report(rows, names, processed, processed * args.every / fps, scale)
    print("\n" + summary)
    (args.out / "summary.txt").write_text(summary + "\n")


if __name__ == "__main__":
    main()
