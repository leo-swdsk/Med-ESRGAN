import argparse
import os
import torch
import warnings

from rrdb_ct_model import RRDBNet_CT


def human_readable(num_params: int) -> str:
    if num_params >= 1_000_000:
        return f"{num_params/1_000_000:.3f} M"
    if num_params >= 1_000:
        return f"{num_params/1_000:.3f} K"
    return str(num_params)


def load_model(scale: int, model_path: str | None) -> torch.nn.Module:
    device = torch.device('cpu')
    model = RRDBNet_CT(scale=scale).to(device)
    if model_path and os.path.isfile(model_path):
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and 'model' in state and all(k in state for k in ['epoch', 'model']):
            state = state['model']
        model.load_state_dict(state, strict=True)
        print(f"[Info] Loaded weights from: {model_path}")
    else:
        if model_path:
            print(f"[Warn] Model path not found or invalid: {model_path}. Counting params of randomly initialized model.")
    model.eval()
    return model


def count_parameters(model: torch.nn.Module) -> tuple[int, int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return total, trainable, non_trainable


def main():
    parser = argparse.ArgumentParser(description='Count parameters of RRDBNet_CT model')
    parser.add_argument('--scale', type=int, default=2, help='Upsampling scale (matches model)')
    parser.add_argument('--model_path', type=str, default=None, help='Optional path to .pth to load before counting')
    parser.add_argument('--profile', action='store_true', help='Also estimate MACs/FLOPs using a dummy LR input')
    parser.add_argument('--input_size', type=int, nargs=2, default=[256, 256], help='LR input size [H W] for MACs/FLOPs (before upscaling)')
    args = parser.parse_args()

    model = load_model(args.scale, args.model_path)
    total, trainable, non_trainable = count_parameters(model)

    print("\nModel: RRDBNet_CT")
    print(f"Scale: x{args.scale}")
    if args.model_path:
        print(f"Weights: {args.model_path}")
    print("---")
    print(f"Total parameters       : {total:,} ({human_readable(total)})")
    print(f"Trainable parameters   : {trainable:,} ({human_readable(trainable)})")
    print(f"Non-trainable parameters: {non_trainable:,} ({human_readable(non_trainable)})")

    if args.profile:
        h, w = int(args.input_size[0]), int(args.input_size[1])
        dummy = torch.randn(1, 1, h, w)
        # Prefer thop; fallback to ptflops; otherwise warn
        macs = None
        try:
            from thop import profile  # type: ignore
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                macs, _ = profile(model, inputs=(dummy,), verbose=False)
        except Exception:
            try:
                # ptflops expects input_res as (C,H,W) and assumes batch=1; it reports MACs by default
                from ptflops import get_model_complexity_info  # type: ignore
                macs_str, _ = get_model_complexity_info(model, (1, h, w), as_strings=True, print_per_layer_stat=False)
                # crude parse, e.g. '12.34 MMac' or '1.23 GMac'
                s = macs_str.lower().replace('mmac', ' m').replace('gmac', ' g').split()
                if len(s) >= 2:
                    val = float(s[0]); unit = s[1]
                    if unit.startswith('g'):
                        macs = int(val * 1_000_000_000)
                    elif unit.startswith('m'):
                        macs = int(val * 1_000_000)
            except Exception:
                macs = None

        def fmt_flops(n: int | None) -> str:
            if n is None:
                return 'N/A'
            if n >= 1_000_000_000:
                return f"{n/1_000_000_000:.3f} G"
            if n >= 1_000_000:
                return f"{n/1_000_000:.3f} M"
            if n >= 1_000:
                return f"{n/1_000:.3f} K"
            return str(n)

        print("---")
        print(f"Profile input (LR)     : 1x1x{h}x{w}")
        if macs is None:
            print("MACs/FLOPs             : N/A (install 'thop' or 'ptflops')")
        else:
            flops = macs * 2  # 1 MAC = 2 FLOPs convention
            print(f"MACs (Multiply-Adds)   : {fmt_flops(macs)}")
            print(f"FLOPs (approx, 2xMACs) : {fmt_flops(flops)}")


if __name__ == '__main__':
    main()


