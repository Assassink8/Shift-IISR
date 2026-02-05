#模型指纹对比

import torch

def get_sd(ckpt):
    """尽量从常见字段里拿到 state_dict，否则就把 ckpt 当 state_dict."""
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "ema"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        # 有些 ckpt 直接就是 {name: tensor, ...}
        if all(isinstance(kk, str) for kk in ckpt.keys()):
            # 只保留 tensor 项
            sd = {k: v for k, v in ckpt.items() if torch.is_tensor(v)}
            if len(sd) > 0:
                return sd
    raise ValueError("Cannot find state_dict in checkpoint. Please adjust get_sd().")

def compare_sd(sd1, sd2, prefix=None, n_show=20):
    """比较两个 state_dict，支持只看某个前缀."""
    if prefix is not None:
        sd1 = {k: v for k, v in sd1.items() if k.startswith(prefix)}
        sd2 = {k: v for k, v in sd2.items() if k.startswith(prefix)}

    k1 = set(sd1.keys())
    k2 = set(sd2.keys())

    missing = sorted(list(k1 - k2))
    unexpected = sorted(list(k2 - k1))
    common = sorted(list(k1 & k2))

    print("=== keys ===")
    print("sd1 keys:", len(k1))
    print("sd2 keys:", len(k2))
    print("common :", len(common))
    print("missing:", len(missing))
    print("unexpected:", len(unexpected))

    if missing:
        print("\n--- missing (sd1 has, sd2 lacks) head ---")
        for k in missing[:n_show]:
            print(k)

    if unexpected:
        print("\n--- unexpected (sd2 has, sd1 lacks) head ---")
        for k in unexpected[:n_show]:
            print(k)

    print("\n=== value diffs (top by max_abs) ===")
    diffs = []
    for k in common:
        v1, v2 = sd1[k], sd2[k]
        if not (torch.is_tensor(v1) and torch.is_tensor(v2)):
            continue
        if v1.shape != v2.shape:
            diffs.append((k, float("inf"), f"shape {tuple(v1.shape)} vs {tuple(v2.shape)}"))
            continue
        a = v1.detach().cpu().float()
        b = v2.detach().cpu().float()
        d = (a - b).abs()
        diffs.append((k, d.max().item(), f"mean_abs={d.mean().item():.6g} dtype1={v1.dtype} dtype2={v2.dtype}"))

    diffs.sort(key=lambda x: x[1], reverse=True)
    for k, max_abs, info in diffs[:n_show]:
        print(f"{k}: max_abs={max_abs:.6g} {info}")

def main():
    ckpt1_path = "/share/huayunpeng-nfs/image_enhancement/ResShift/weights/resshift_bicsrx4_s4.pth"   # 改成你的路径
    ckpt2_path = "saved.pt"  # 改成你的路径

    ckpt1 = torch.load(ckpt1_path, map_location="cpu")
    ckpt2 = torch.load(ckpt2_path, map_location="cpu")

    sd1 = get_sd(ckpt1)
    sd2 = get_sd(ckpt2)

    # 如果你只想看 UNet 的部分，比如 key 以 'unet.' 开头，就填 prefix='unet.'
    # 不确定前缀就先 prefix=None 看全量，再自己观察 key 命名。
    compare_sd(sd1, sd2, prefix=None, n_show=30)

if __name__ == "__main__":
    main()
