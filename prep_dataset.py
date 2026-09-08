#!/usr/bin/env python3
"""Build a YOLO detection dataset for RoadWatch out of RDD2022ES plus a manhole source.

RDD2022ES ships as one flat directory of images and YOLO .txt files covering eight
severity-split damage classes. We keep the two pothole tiers, fold in manhole boxes
from a second dataset so the model learns to leave storm drains alone, and emit the
train/val/test layout Ultralytics expects.
"""

import argparse
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_NAMES = ["pothole", "pothole_deep", "manhole"]
POTHOLE, POTHOLE_DEEP, MANHOLE = 0, 1, 2

RDD_REMAP = {6: POTHOLE, 7: POTHOLE_DEEP}
MANHOLE_REMAP = {2: MANHOLE}
MANHOLE_SOURCE_POTHOLE = 0

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def resolve_root(requested, marker, search=Path("/kaggle/input")):
    """Locate a dataset directory when the mount name differs from what we assumed.

    Kaggle names each mount after the dataset slug, which does not always match what
    a notebook was written against. Rather than fail on a hardcoded guess, look for
    the directory by a marker we know it contains, and say what was actually there
    when that fails too.
    """
    if requested.is_dir():
        return requested

    candidates = sorted(search.glob(marker)) if search.is_dir() else []
    if len(candidates) == 1:
        print(f"{requested} missing, resolved to {candidates[0]}")
        return candidates[0]

    listing = sorted(p.name for p in search.iterdir()) if search.is_dir() else []
    raise SystemExit(
        f"{requested} not found and {marker!r} matched {len(candidates)} directories "
        f"{[str(c) for c in candidates]}. Contents of {search}: {listing}"
    )


def thumbnails(paths, size=16):
    """Contrast-normalised grayscale thumbnails, so JPEG quality differences wash out.

    draft() lets libjpeg decode straight to a small scale, which is the difference
    between this pass taking seconds and taking many minutes.
    """
    thumbs = np.empty((len(paths), size, size), dtype=np.float32)
    for i, path in enumerate(paths):
        with Image.open(path) as img:
            img.draft("L", (size * 4, size * 4))
            arr = np.asarray(img.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32)
        thumbs[i] = (arr - arr.mean()) / (arr.std() + 1e-6)
    return thumbs


def drop_mirrors(paths, threshold=0.85, chunk=512):
    """Drop images that are horizontal flips of another image already in the set.

    RDD2022ES ships roughly half its frames as mirrored copies of the other half.
    Ultralytics already applies fliplr during training, so keeping both halves buys
    nothing and doubles the epoch. Matching is a cosine similarity between each
    thumbnail and every other thumbnail's mirror; a pair has to be each other's
    mutual best match to count, which keeps near-identical frames from chaining.
    """
    if len(paths) < 2:
        return list(paths), 0

    thumbs = thumbnails(paths)
    straight = thumbs.reshape(len(paths), -1)
    mirrored = thumbs[:, :, ::-1].reshape(len(paths), -1)
    scale = straight.shape[1]

    best = np.empty(len(paths), dtype=np.int64)
    score = np.empty(len(paths), dtype=np.float32)
    for start in range(0, len(paths), chunk):
        stop = min(start + chunk, len(paths))
        sims = straight[start:stop] @ mirrored.T / scale
        sims[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        best[start:stop] = sims.argmax(axis=1)
        score[start:stop] = sims.max(axis=1)

    dropped = set()
    for i, j in enumerate(best):
        if score[i] < threshold or best[j] != i:
            continue
        dropped.add(max(i, j))
    return [p for i, p in enumerate(paths) if i not in dropped], len(dropped)


def remap_lines(label_path, table):
    if not label_path.exists():
        return []
    lines = []
    for raw in label_path.read_text().splitlines():
        parts = raw.split()
        if not parts:
            continue
        source_class = int(float(parts[0]))
        if source_class in table:
            lines.append(" ".join([str(table[source_class])] + parts[1:]))
    return lines


def country_of(path):
    head, _, tail = path.stem.rpartition("_")
    return head if head and tail.isdigit() else "unknown"


def load_rdd(root, countries, keep_mirrors, mirror_threshold):
    root = resolve_root(root, "*/combined_annotatedv2")
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit(f"no images found under {root}")

    found = Counter(country_of(p) for p in images)
    print(f"RDD countries present: {dict(found)}")
    if countries:
        images = [p for p in images if country_of(p) in countries]
        print(f"kept {len(images)} images after country filter {countries}")

    if keep_mirrors:
        dropped = 0
    else:
        images, dropped = drop_mirrors(images, mirror_threshold)
    print(f"RDD: {len(images)} images kept, {dropped} mirrored duplicates dropped")

    return [(p, remap_lines(p.with_suffix(".txt"), RDD_REMAP)) for p in images]


def load_manhole(root):
    """Collect manhole boxes, keeping only frames that contain no pothole.

    This dataset ships the same images under all_classes/ and under per-class
    folders, so one stem can carry two label files holding different subsets of the
    truth; taking the union of their lines is correct whichever layout we walk into.
    Frames that also hold a pothole are dropped rather than partially labelled. The
    source has no severity annotation so its potholes cannot be placed in either of
    our two tiers, and keeping the frame with the pothole unmarked would train the
    model to read a real pothole as background.
    """
    root = resolve_root(root, "*manhole*")

    images = {}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in IMAGE_EXTS:
            images.setdefault(path.stem, path)

    boxes = defaultdict(set)
    contaminated = set()
    for label_path in sorted(root.rglob("*.txt")):
        stem = label_path.stem
        if stem not in images:
            continue
        for raw in label_path.read_text().splitlines():
            parts = raw.split()
            if not parts:
                continue
            source_class = int(float(parts[0]))
            if source_class == MANHOLE_SOURCE_POTHOLE:
                contaminated.add(stem)
            elif source_class in MANHOLE_REMAP:
                boxes[stem].add(" ".join([str(MANHOLE_REMAP[source_class])] + parts[1:]))

    records = [(images[stem], sorted(lines)) for stem, lines in sorted(boxes.items())
               if lines and stem not in contaminated]
    skipped = len([s for s in boxes if s in contaminated])
    print(f"manhole: {len(records)} images kept, {skipped} skipped for holding unlabelled potholes")
    return records


def split(records, val_frac, test_frac, seed):
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    return {
        "val": shuffled[:n_val],
        "test": shuffled[n_val:n_val + n_test],
        "train": shuffled[n_val + n_test:],
    }


def write_split(out, name, records, prefix, copy):
    image_dir = out / "images" / name
    label_dir = out / "labels" / name
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    for image_path, lines in records:
        stem = f"{prefix}{image_path.stem}"
        target = image_dir / f"{stem}{image_path.suffix.lower()}"
        if copy:
            target.write_bytes(image_path.read_bytes())
        elif not target.exists():
            os.symlink(image_path.resolve(), target)
        (label_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        for line in lines:
            counts[int(line.split()[0])] += 1
        if not lines:
            counts["background"] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rdd", type=Path, required=True,
                        help="RDD2022ES combined_annotatedv2 directory")
    parser.add_argument("--manhole", type=Path,
                        help="root of sabidrahman/pothole-cracks-and-openmanhole")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--countries", nargs="*", default=None,
                        help="filename prefixes to keep, e.g. United_States Czech")
    parser.add_argument("--background-frac", type=float, default=0.1,
                        help="share of final training images that carry no boxes")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-mirrors", action="store_true",
                        help="skip mirror detection (RDD2022ES ships flipped duplicates)")
    parser.add_argument("--mirror-threshold", type=float, default=0.85,
                        help="cosine similarity above which two frames count as a mirrored pair")
    parser.add_argument("--copy", action="store_true",
                        help="copy images instead of symlinking into the source tree")
    args = parser.parse_args()

    rdd = load_rdd(args.rdd, args.countries, args.keep_mirrors, args.mirror_threshold)
    labelled = [r for r in rdd if r[1]]
    background = [r for r in rdd if not r[1]]

    # Backgrounds suppress false positives, but RDD is mostly empty frames and
    # letting them dominate teaches the model that predicting nothing is safe.
    budget = int(len(labelled) * args.background_frac / max(1e-9, 1 - args.background_frac))
    random.Random(args.seed).shuffle(background)
    background = background[:budget]
    print(f"RDD: {len(labelled)} labelled, {len(background)} background frames retained")

    sources = [("rdd_", labelled + background)]
    if args.manhole:
        sources.append(("mh_", load_manhole(args.manhole)))

    if args.out.exists():
        raise SystemExit(f"{args.out} already exists, remove it first")

    totals = defaultdict(Counter)
    for prefix, records in sources:
        for name, chunk in split(records, args.val_frac, args.test_frac, args.seed).items():
            totals[name] += write_split(args.out, name, chunk, prefix, args.copy)

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n\n"
        f"names:\n{names}\n"
    )

    for name in ("train", "val", "test"):
        counts = totals[name]
        summary = ", ".join(f"{CLASS_NAMES[i]}={counts[i]}" for i in range(len(CLASS_NAMES)))
        n_images = len(list((args.out / "images" / name).iterdir()))
        print(f"{name}: {n_images} images, {summary}, background={counts['background']}")


if __name__ == "__main__":
    main()
