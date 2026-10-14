import sys, torch
from memtree.distill import load_model
from memtree.phi_adapter import load_adapter, teacher_mode

model, tok = load_model("richardyoung/Phi-4-mini-instruct-heretic", "cuda", torch.bfloat16)
load_adapter(model, sys.argv[1] if len(sys.argv) > 1 else "adapter_quick.pt")

def reply(msgs):
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    ids = (ids if hasattr(ids, "shape") else ids["input_ids"]).to("cuda")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=200, do_sample=False)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

history = []
while True:
    q = input("\nyou> ").strip()
    if q in ("quit", "exit"):
        break
    history.append({"role": "user", "content": q})
    a = reply(history)
    print("\ntree phi>", a)
    with teacher_mode(model):
        print("\noriginal phi>", reply(history))
    history.append({"role": "assistant", "content": a})
