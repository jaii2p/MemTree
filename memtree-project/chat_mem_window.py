"""Chat with linear-context Phi + the watchdog (no tree inside her attention).

    python chat_mem_window.py finetune_v4.pt 1024
Commands: /mem shows what the watchdog puts in front of her; quit exits.
"""
import sys
import torch
from memtree.distill import load_model
from memtree.finetune import load_finetune
from memtree.phi_adapter import window_model
from memtree.runtime import MemoryRuntime, render_memory

ft = sys.argv[1] if len(sys.argv) > 1 else "finetune_v4.pt"
window = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
model, tok = load_model("richardyoung/Phi-4-mini-instruct-heretic", device, dtype)
window_model(model, window=window)
load_finetune(model, tok, ft)
rt = MemoryRuntime(model, tok, window=window)

while True:
    q = input("\nyou> ").strip()
    if q in ("quit", "exit"):
        break
    if q == "/mem":
        print(render_memory(rt.dog, "me", entry_chars=rt.budget["entry_chars"]))
        continue
    r = rt.chat("me", q, max_new_tokens=400)
    print("\nphi>", r["reply"])
    for d in r["references"]:
        print(f"  [watchdog] {d.ref[1]} -> {d.resolution}")
