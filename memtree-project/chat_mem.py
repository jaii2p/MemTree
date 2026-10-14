import sys, torch
from memtree.distill import load_model
from memtree.phi_adapter import load_adapter
from memtree.finetune import load_finetune
from memtree.runtime import MemoryRuntime, render_memory

model, tok = load_model("richardyoung/Phi-4-mini-instruct-heretic", "cuda", torch.bfloat16)
load_adapter(model, "adapter_v2.pt")
load_finetune(model, tok, "finetune_v2.pt")
rt = MemoryRuntime(model, tok)

while True:
    q = input("\nyou> ").strip()
    if q in ("quit", "exit"):
        break
    if q == "/mem":
        print(render_memory(rt.dog, "me"))
        continue
    r = rt.chat("me", q)
    print("\nphi>", r["reply"])
    for d in r["references"]:
        print(f"  [watchdog] {d.ref[1]} -> {d.resolution}")
