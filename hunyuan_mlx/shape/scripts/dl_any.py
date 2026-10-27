"""Generic resumable ModelScope downloader. Reads exact size from the repo API.
Usage: python scripts/dl_any.py <repo> <filepath> <outpath>
"""

import argparse
import json
import os
import sys
import time
import urllib.request

TERRITORIAL = (
    "These are Tencent Hunyuan Community License weights, not licensed for use in "
    "the EU, the UK or South Korea."
)

API = "https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true"
BASE = "https://modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={path}"


def _opener():
    return urllib.request.build_opener(
        urllib.request.ProxyHandler(urllib.request.getproxies())
    )


def lookup_size(repo, filepath):
    """Byte size from the repo file list. Not the weight file itself."""
    size = None
    with _opener().open(API.format(repo=repo), timeout=60) as response:
        for entry in json.load(response)["Data"]["Files"]:
            if entry.get("Path") == filepath:
                size = entry["Size"]
    return size


def download_file(repo, filepath, out):
    size = lookup_size(repo, filepath)
    base = BASE.format(repo=repo, path=filepath)
    print(f"target {filepath}: {size} bytes ({size / 1e9:.2f} GB)", flush=True)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    net = _opener()

    for attempt in range(1, 81):
        have = os.path.getsize(out) if os.path.exists(out) else 0
        if have >= size:
            print(f"complete ({have} bytes)", flush=True)
            break
        req = urllib.request.Request(base)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            t0 = time.time()
            with net.open(req, timeout=120) as response, open(out, "ab") as handle:
                got = 0
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
                    got += len(chunk)
                    if got % (256 << 20) < (1 << 20):
                        speed = got / 1e6 / max(1e-3, time.time() - t0)
                        print(
                            f"  {(have + got) / 1e9:.2f} GB ({speed:.1f} MB/s)",
                            flush=True,
                        )
        except urllib.error.HTTPError as exc:
            if exc.code == 416:
                print("complete (416)", flush=True)
                break
            print(f"  attempt {attempt}: HTTP {exc.code}", flush=True)
            time.sleep(min(20, 2 ** (attempt % 6)))
        except Exception as exc:  # noqa: BLE001
            print(
                f"  attempt {attempt}: {type(exc).__name__} {str(exc)[:80]}", flush=True
            )
            time.sleep(min(20, 2 ** (attempt % 6)))
    done = os.path.exists(out) and os.path.getsize(out) >= size
    print("DONE" if done else "INCOMPLETE", flush=True)
    return done


def confirmed(yes: bool) -> bool:
    if yes:
        return True
    if not sys.stdin or not sys.stdin.isatty():
        print("Refusing to download without --yes when there is nobody to ask.")
        return False
    try:
        answer = input("Continue? [y/N] ")
    except EOFError:
        print("Refusing to download without --yes when there is nobody to ask.")
        return False
    if answer.strip().lower() not in {"y", "yes"}:
        print("Nothing downloaded.")
        return False
    return True


def announce(repo: str, filepath: str, out: str, size: int) -> None:
    print("Downloading Hunyuan weights from ModelScope:", flush=True)
    print(f"  repo: {repo}", flush=True)
    print(f"  file: {filepath}", flush=True)
    print(f"  out:  {out}", flush=True)
    print(f"  size: {size} bytes ({size / 1e9:.2f} GB)", flush=True)
    print(TERRITORIAL, flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("repo")
    parser.add_argument("filepath")
    parser.add_argument("outpath")
    parser.add_argument(
        "--yes", action="store_true", help="download without asking again"
    )
    args = parser.parse_args(argv)
    size = lookup_size(args.repo, args.filepath)
    if not isinstance(size, int) or size < 0:
        print("That file is not in the ModelScope listing. Nothing downloaded.")
        return 1
    announce(args.repo, args.filepath, args.outpath, size)
    if not confirmed(args.yes):
        return 1
    return 0 if download_file(args.repo, args.filepath, args.outpath) else 1


if __name__ == "__main__":
    raise SystemExit(main())
