"""Convert the v2-1 .ckpt (torch pickle) -> flat fp16 safetensors with
conditioner./model./vae. prefixes that hy3dmlx.convert.load_models expects.

The released inference ckpt is a dict {'model':sd, 'vae':sd, 'conditioner':sd}.
Torch is not installed in the shape environment. Run this with the oracle
interpreter, and pass the checkpoint path and the safetensors path:

  .venv-oracle/bin/python scripts/convert_v21_ckpt.py <ckpt> <safetensors>
"""

import argparse
import collections


def convert(ckpt_path: str, out_path: str) -> None:
    import torch

    print(">> torch.load (cpu)...", flush=True)
    # weights_only=True refuses a pickle. Do not retry with False: that would
    # execute whatever a re-uploaded checkpoint contains.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if isinstance(ckpt, dict) and "state_dict" in ckpt and "model" not in ckpt:
        ckpt = ckpt["state_dict"]

    flat = {}
    subs = [
        k
        for k in ("model", "vae", "conditioner")
        if k in ckpt and isinstance(ckpt[k], dict)
    ]
    if subs:  # dict-of-substatedicts layout
        print(f"  layout: dict of sub-state-dicts {subs}", flush=True)
        for sub in subs:
            for k, v in ckpt[sub].items():
                flat[f"{sub}.{k}"] = v
    else:  # already-flat layout, possibly with a wrapper prefix
        print("  layout: flat", flush=True)
        flat = {k: v for k, v in ckpt.items()}

    # fp16 floats, contiguous
    out = {}
    for k, v in flat.items():
        if not torch.is_tensor(v):
            continue
        out[k] = (v.half() if v.is_floating_point() else v).contiguous()

    pref = collections.Counter(k.split(".")[0] for k in out)
    print("  top-level prefixes:", dict(pref), flush=True)
    print("  total tensors:", len(out), flush=True)
    import safetensors.torch as save_safetensors

    save_safetensors.save_file(out, out_path)
    print(f"WROTE {out_path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ckpt", help="path to the downloaded model.fp16.ckpt")
    parser.add_argument("out", help="path to write model.fp16.safetensors")
    args = parser.parse_args(argv)
    convert(args.ckpt, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
