import gc
import gzip
import hashlib
import json
import os
import struct
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch import nn

SEED = 20260929
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))

ROOT = Path(os.environ.get("FLY_REPO", "/tmp/flybrain"))
OUT = Path(os.environ.get("FLY_OUT", "flybrain-hybrid-results"))
OUT.mkdir(parents=True, exist_ok=True)
CONNECTOME = ROOT / "data" / "connectome.bin.gz"
META_JSON = ROOT / "data" / "neuron_meta.json"
LLM_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
SD_PRIMARY = "segmind/tiny-sd"
SD_FALLBACK = "diffusers/tiny-stable-diffusion-torch"


def log(msg):
    print(f"[flybrain-hybrid] {msg}", flush=True)


def parse_connectome(path: Path):
    raw_gz = path.read_bytes()
    raw = gzip.decompress(raw_gz)
    n, e = struct.unpack_from("<II", raw, 0)
    edge_dtype = np.dtype([("pre", "<u4"), ("post", "<u4"), ("w", "<f4")])
    edges = np.frombuffer(raw, dtype=edge_dtype, count=e, offset=8)
    meta_offset = 8 + e * 12
    neuron_dtype = np.dtype([("region", "u1"), ("group", "<u2")])
    nm = np.frombuffer(raw, dtype=neuron_dtype, count=n, offset=meta_offset)

    pre = edges["pre"].astype(np.int32, copy=False)
    post = edges["post"].astype(np.int32, copy=False)
    w = edges["w"].astype(np.float32, copy=True)
    mx = float(np.max(np.abs(w))) if e else 1.0
    w *= 0.15 / max(mx, 1e-12)
    W = csr_matrix((w, (pre, post)), shape=(n, n), dtype=np.float32)
    gids = nm["group"].astype(np.int32, copy=False)
    sha = hashlib.sha256(raw_gz).hexdigest()
    return n, e, W, gids, sha


def simulate(W, gids, stimulus_groups, intensity=0.15, ticks=100):
    n = W.shape[0]
    V = np.zeros(n, np.float32)
    fired = np.zeros(n, np.float32)
    refractory = np.zeros(n, np.uint8)
    stim = np.flatnonzero(np.isin(gids, np.asarray(stimulus_groups, dtype=np.int32)))
    group_hist = np.zeros((ticks, 63), np.float32)

    for t in range(ticks):
        ref = refractory > 0
        refractory[ref] -= 1
        V[ref] = 0.0
        V[~ref] *= 0.95
        if fired.any():
            V += W.T.dot(fired).astype(np.float32, copy=False)
        active_stim = stim[refractory[stim] == 0]
        V[active_stim] += intensity
        new = (refractory == 0) & (V >= 1.0)
        fired = new.astype(np.float32)
        if new.any():
            V[new] = 0.0
            refractory[new] = 3
            group_hist[t] = np.bincount(gids[new], minlength=63)[:63]

    feature = np.log1p(group_hist[ticks // 2 :].sum(axis=0)).astype(np.float32)
    feature /= float(np.linalg.norm(feature) + 1e-8)
    return feature, group_hist, int(stim.size)


class BrainPromptAdapter(nn.Module):
    def __init__(self, dim=63, classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, classes),
        )

    def forward(self, x):
        return self.net(x)


def train_adapter(X):
    # Augment the four real connectome states with small measurement noise.
    rng = np.random.default_rng(SEED)
    xx, yy = [], []
    for i, row in enumerate(X):
        for _ in range(192):
            z = row + rng.normal(0.0, 0.025, row.shape).astype(np.float32)
            z = np.clip(z, 0, None)
            z /= np.linalg.norm(z) + 1e-8
            xx.append(z)
            yy.append(i)
    xx = torch.tensor(np.stack(xx), dtype=torch.float32)
    yy = torch.tensor(yy, dtype=torch.long)

    model = BrainPromptAdapter()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    trace = []
    for step in range(350):
        logits = model(xx)
        loss = criterion(logits, yy)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step in {0, 1, 2, 4, 9, 24, 49, 99, 199, 349}:
            acc = float((logits.argmax(-1) == yy).float().mean())
            trace.append({"step": step + 1, "loss": float(loss.detach()), "accuracy": acc})
            log(f"adapter step={step+1} loss={loss.item():.6f} acc={acc:.4f}")

    with torch.no_grad():
        real_logits = model(torch.tensor(X, dtype=torch.float32))
        probs = torch.softmax(real_logits, -1).cpu().numpy()
        pred = real_logits.argmax(-1).cpu().numpy()
    return model, trace, probs, pred


def run_brain():
    if not CONNECTOME.exists():
        raise FileNotFoundError(f"missing {CONNECTOME}")
    n, e, W, gids, sha = parse_connectome(CONNECTOME)
    meta = json.loads(META_JSON.read_text(encoding="utf-8")) if META_JSON.exists() else None
    log(f"parsed connectome: {n:,} neurons, {e:,} edges, sha256={sha}")

    scenarios = {
        "sweet_food": [32],
        "visual_light": [0, 2],
        "touch": [10],
        "warm": [14],
    }
    names = list(scenarios)
    X = []
    report = {
        "neuron_count": n,
        "edge_count": e,
        "connectome_sha256": sha,
        "lif_constants": {"leak": 0.95, "threshold": 1.0, "refractory_ticks": 3, "weight_scale": 0.15, "stimulus_intensity": 0.15},
        "states": {},
    }
    for name, groups in scenarios.items():
        t0 = time.time()
        feat, hist, nstim = simulate(W, gids, groups)
        X.append(feat)
        top_ids = np.argsort(feat)[::-1][:8]
        top = []
        for gid in top_ids:
            if feat[gid] <= 0:
                continue
            gname = str(gid)
            if meta and gid < len(meta.get("groups", [])):
                gname = meta["groups"][int(gid)]["name"]
            top.append({"group_id": int(gid), "name": gname, "feature": float(feat[gid])})
        report["states"][name] = {
            "stimulus_group_ids": groups,
            "stimulated_neurons": nstim,
            "total_spikes": int(hist.sum()),
            "top_active_groups": top,
            "elapsed_sec": round(time.time() - t0, 3),
        }
        log(f"brain state={name} stim_neurons={nstim:,} spikes={int(hist.sum()):,} top={top[:4]}")

    X = np.stack(X).astype(np.float32)
    report["feature_cosine_similarity"] = np.round(X @ X.T, 6).tolist()
    (OUT / "brain_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "brain_states.npz", names=np.array(names), X=X)
    del W
    gc.collect()
    return X, names, report


def run_llm(X, names, adapter, probs):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    policies = {
        "sweet_food": "The fly brain is responding to sweet food and feeding-related sensory input.",
        "visual_light": "The fly brain is receiving strong visual input and orienting toward light.",
        "touch": "The fly brain is responding to sudden mechanosensory touch and a startle-like event.",
        "warm": "The fly brain is responding to warm temperature sensory input.",
    }
    log(f"loading LLM {LLM_ID}")
    tok = AutoTokenizer.from_pretrained(LLM_ID)
    model = AutoModelForCausalLM.from_pretrained(LLM_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()

    outputs = {}
    for i, source_name in enumerate(names):
        decoded_idx = int(np.argmax(probs[i]))
        decoded_name = names[decoded_idx]
        top_groups = json.loads((OUT / "brain_results.json").read_text())["states"][source_name]["top_active_groups"][:4]
        group_text = ", ".join(g["name"] for g in top_groups)
        user_text = (
            f"You are interpreting a simulated Drosophila connectome state. {policies[decoded_name]} "
            f"The most active functional groups are: {group_text}. "
            "Describe the current state in one concise factual sentence."
        )
        messages = [{"role": "user", "content": user_text}]
        input_ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=42,
                do_sample=False,
                repetition_penalty=1.08,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(generated[0, input_ids.shape[-1]:], skip_special_tokens=True).strip()
        outputs[source_name] = {
            "adapter_decoded_state": decoded_name,
            "adapter_probabilities": {names[j]: float(probs[i, j]) for j in range(len(names))},
            "prompt": user_text,
            "output": text,
        }
        log(f"LLM {source_name} -> {decoded_name}: {text}")

    report = {"model": LLM_ID, "backbone_frozen": True, "outputs": outputs}
    (OUT / "llm_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    del model, tok
    gc.collect()
    return report


def load_sd():
    from diffusers import StableDiffusionPipeline
    err = None
    try:
        log(f"loading Stable Diffusion {SD_PRIMARY}")
        pipe = StableDiffusionPipeline.from_pretrained(SD_PRIMARY, torch_dtype=torch.float32, safety_checker=None)
        return pipe, SD_PRIMARY, err
    except Exception as exc:
        err = repr(exc)
        log(f"primary SD failed: {err}")
        log(f"loading fallback Stable Diffusion {SD_FALLBACK}")
        pipe = StableDiffusionPipeline.from_pretrained(SD_FALLBACK, torch_dtype=torch.float32, safety_checker=None)
        return pipe, SD_FALLBACK, err


def run_sd(X, names, probs):
    prompts = {
        "sweet_food": "macro photograph of a fruit fly feeding on ripe red fruit, detailed natural light, realistic biology",
        "visual_light": "macro photograph of a fruit fly oriented toward a bright light, realistic insect, dramatic illumination",
        "touch": "macro photograph of a startled fruit fly reacting to sudden touch, realistic insect, fast movement",
        "warm": "macro photograph of a fruit fly in a warm amber environment, realistic insect, natural light",
    }
    pipe, model_id, primary_error = load_sd()
    pipe = pipe.to("cpu")
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=False)

    generated = {}
    # Generate two distinct connectome-conditioned states to verify the adapter changes generation.
    for source_name in ("sweet_food", "touch"):
        i = names.index(source_name)
        decoded_idx = int(np.argmax(probs[i]))
        decoded_name = names[decoded_idx]
        prompt = prompts[decoded_name]
        gen = torch.Generator(device="cpu").manual_seed(SEED + i)
        t0 = time.time()
        with torch.no_grad():
            image = pipe(
                prompt,
                num_inference_steps=6 if model_id == SD_PRIMARY else 4,
                guidance_scale=5.5,
                height=256,
                width=256,
                generator=gen,
            ).images[0]
        path = OUT / f"sd_{source_name}.png"
        image.save(path)
        generated[source_name] = {
            "adapter_decoded_state": decoded_name,
            "prompt": prompt,
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": [image.width, image.height],
            "elapsed_sec": round(time.time() - t0, 3),
        }
        log(f"SD {source_name} -> {decoded_name}: {path} {generated[source_name]['sha256'][:16]}")

    report = {
        "model_requested": SD_PRIMARY,
        "model_used": model_id,
        "primary_load_error": primary_error,
        "backbone_frozen": True,
        "generated": generated,
    }
    (OUT / "sd_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    del pipe
    gc.collect()
    return report


def main():
    t0 = time.time()
    X, names, brain = run_brain()
    adapter, trace, probs, pred = train_adapter(X)
    torch.save(adapter.state_dict(), OUT / "brain_prompt_adapter.pt")
    adapter_report = {
        "architecture": "63 -> 64 -> 32 -> 4",
        "training": "cross-entropy on noisy augmentations of four real connectome activity states",
        "trace": trace,
        "real_state_probabilities": {names[i]: {names[j]: float(probs[i, j]) for j in range(len(names))} for i in range(len(names))},
        "real_state_predictions": {names[i]: names[int(pred[i])] for i in range(len(names))},
    }
    (OUT / "adapter_results.json").write_text(json.dumps(adapter_report, indent=2), encoding="utf-8")

    llm = run_llm(X, names, adapter, probs)
    sd = run_sd(X, names, probs)

    upstream_commit = os.environ.get("FLY_UPSTREAM_COMMIT", "unknown")
    summary = {
        "seed": SEED,
        "flybrain_upstream_commit": upstream_commit,
        "neuron_count": brain["neuron_count"],
        "edge_count": brain["edge_count"],
        "connectome_sha256": brain["connectome_sha256"],
        "adapter_final": trace[-1],
        "llm_model": llm["model"],
        "sd_model_used": sd["model_used"],
        "elapsed_sec": round(time.time() - t0, 3),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# FlyBrain × SmolLM2 × Stable Diffusion actual run",
        "",
        f"- FlyBrain upstream commit: `{upstream_commit}`",
        f"- Connectome: **{brain['neuron_count']:,} neurons / {brain['edge_count']:,} edges**",
        f"- Connectome SHA-256: `{brain['connectome_sha256']}`",
        f"- Brain adapter: `63 → 64 → 32 → 4`, final train accuracy `{trace[-1]['accuracy']:.4f}`",
        f"- LLM: `{llm['model']}`",
        f"- Stable Diffusion: `{sd['model_used']}`",
        f"- Total elapsed: `{summary['elapsed_sec']} s`",
        "",
        "## LLM outputs",
    ]
    for name, item in llm["outputs"].items():
        lines.append(f"- **{name}** → **{item['adapter_decoded_state']}**: {item['output']}")
    lines += ["", "## Stable Diffusion outputs"]
    for name, item in sd["generated"].items():
        lines.append(f"- **{name}** → **{item['adapter_decoded_state']}**: `{item['file']}` · sha256 `{item['sha256']}`")
    lines += [
        "",
        "## Training scope",
        "The biological connectome is executed as an LIF network. A small neural adapter is actually trained on its 63-dimensional activity states. The open-source LLM and Stable Diffusion backbones are kept frozen and are conditioned by the trained adapter. This is real adapter training and real inference; it is not a claim that the fly connectome itself has been converted into an LLM or diffusion UNet.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("DONE " + json.dumps(summary))


if __name__ == "__main__":
    main()
