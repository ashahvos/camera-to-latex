"""Download public datasets into data/downloads/ (optional: synthetic data + sheets are enough for v1).

  python -m tools.download_datasets --list
  python -m tools.download_datasets chars74k_fnt hasy
  python -m tools.download_datasets tmnist94          # needs a Kaggle account + API token

Then convert them to crops with:  python -m tools.import_datasets <name>

Which ones help, and why (details in the instructions PDF):
  v1 symbol classifier (printed symbols, one at a time)
    chars74k_fnt  printed digits and letters in many computer fonts      -> recommended
    tmnist94      all 94 printable ASCII characters in many Google Fonts -> recommended
    hasy          handwritten LaTeX symbols, 369 classes                 -> optional, small share
    kaggle_math   handwritten math symbols (from CROHME)                 -> optional, small share
  revision 2 (whole-line image -> LaTeX model, handwriting)
    im2latex      rendered formulas with their LaTeX source
    mathwriting   handwritten math expressions (Google), large
    crohme2023    handwritten math expressions (competition data)
Licences differ (some are non-commercial): read the page linked for each one
before using it, and cite it in the report.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

DATASETS = {
    "chars74k_fnt": {
        "what": "Chars74K EnglishFnt: 62,992 characters (0-9, A-Z, a-z) from computer fonts, 128x128 PNG",
        "url": "http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz",
        "page": "http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/",
        "size": "see page", "use": "v1", "import": True},
    "tmnist94": {
        "what": "TMNIST, 94-character subset: every printable ASCII glyph in many Google Fonts, 28x28",
        "kaggle": "nikbearbrown/tmnist-alphabet-94-characters",
        "page": "https://www.kaggle.com/datasets/nikbearbrown/tmnist-alphabet-94-characters",
        "size": "see page", "use": "v1", "import": True},
    "hasy": {
        "what": "HASYv2: 168,233 handwritten symbols, 369 LaTeX classes, 32x32 PNG",
        "url": "https://zenodo.org/records/259444/files/HASYv2.tar.bz2?download=1",
        "file": "HASYv2.tar.bz2",
        "page": "https://zenodo.org/records/259444",
        "size": "35 MB", "use": "v1 (optional)", "import": True},
    "kaggle_math": {
        "what": "Handwritten math symbols: 82 classes, 45x45 JPG, extracted from CROHME",
        "kaggle": "xainano/handwrittenmathsymbols",
        "page": "https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols",
        "size": "see page", "use": "v1 (optional)", "import": True},
    "im2latex": {
        "what": "IM2LATEX-100K: 103,556 rendered formulas with their LaTeX",
        "page": "https://zenodo.org/records/56198",
        "size": "about 300 MB", "use": "revision 2", "import": False},
    "mathwriting": {
        "what": "MathWriting: 230k human + about 400k synthetic handwritten expressions (InkML)",
        "url": "https://storage.googleapis.com/mathwriting_data/mathwriting-2024-excerpt.tgz",
        "file": "mathwriting-2024-excerpt.tgz",
        "full": "https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz (2.9 GB)",
        "page": "https://github.com/google-research/google-research/tree/master/mathwriting",
        "size": "excerpt 1.5 MB (full set 2.9 GB)", "use": "revision 2", "import": False},
    "crohme2023": {
        "what": "CROHME 2023: handwritten math expressions, online + offline",
        "page": "https://zenodo.org/records/8428035",
        "size": "1.8 GB", "use": "revision 2", "import": False},
}


def _progress(block, size, total):
    if total > 0:
        done = min(total, block * size)
        print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)


def _extract(archive: Path, dest: Path) -> None:
    """Unpack into dest, refusing paths that would escape it."""
    dest = dest.resolve()
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            for m in z.namelist():
                if not (dest / m).resolve().is_relative_to(dest):
                    raise RuntimeError(f"unsafe path in {archive.name}: {m}")
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            try:
                t.extractall(dest, filter="data")
            except TypeError:  # Python < 3.12
                for m in t.getmembers():
                    if not (dest / m.name).resolve().is_relative_to(dest):
                        raise RuntimeError(f"unsafe path in {archive.name}: {m.name}")
                t.extractall(dest)


def download(name: str, root: Path) -> bool:
    d = DATASETS[name]
    dest = root / name
    dest.mkdir(parents=True, exist_ok=True)
    if "kaggle" in d:
        exe = shutil.which("kaggle")
        if exe is None:
            print(f"{name}: needs the Kaggle CLI (pip install kaggle, then put your API token in "
                  f"~/.kaggle/kaggle.json), or download it by hand from {d['page']} and unzip into {dest}")
            return False
        r = subprocess.run([exe, "datasets", "download", "-d", d["kaggle"], "-p", str(dest), "--unzip"])
        return r.returncode == 0
    if "url" not in d:
        print(f"{name}: download by hand from {d['page']} into {dest}")
        return False
    archive = dest / d.get("file", d["url"].rsplit("/", 1)[-1])
    if not archive.exists():
        print(f"{name}: downloading {d['url']}")
        tmp = archive.with_suffix(archive.suffix + ".part")
        urllib.request.urlretrieve(d["url"], tmp, _progress)
        tmp.replace(archive)
        print()
    print(f"{name}: extracting {archive.name}")
    _extract(archive, dest)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="datasets to download")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--root", type=Path, default=Path(config.DOWNLOADS_DIR))
    args = ap.parse_args(argv)
    if args.list or not args.names:
        for k, d in DATASETS.items():
            how = "kaggle" if "kaggle" in d else ("url" if "url" in d else "by hand")
            print(f"{k:13s} [{d['use']}, {d['size']}, {how}]\n              {d['what']}\n              {d['page']}")
        return 0
    ok = True
    for n in args.names:
        if n not in DATASETS:
            print(f"unknown dataset {n}; see --list")
            ok = False
            continue
        ok = download(n, args.root) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
