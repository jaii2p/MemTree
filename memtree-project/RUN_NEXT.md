# Next GPU session: tree Phi + living memory tree

Phi keeps her tree attention exactly as built (adapter_v3.pt, or adapter_v2.pt).
Messages stay untouched; overflow is copied onto the memory tree, which the
watchdog tends. About 25 min on one GPU. Design notes: DESIGN.md

    unzip -o memtree_project.zip && cd memtree-project
    pip install -r requirements.txt
    python run_tests.py                                  # expect 0 failed

    # 1. dataset (reuses tonight's Phi-voiced answers; corpus.txt for passages)
    python scripts/make_corpus.py --tokens 5M            # only if corpus.txt is missing
    python scripts/make_chat_sft.py --model richardyoung/Phi-4-mini-instruct-heretic \
        --filler-from chat_sft.jsonl.filler.json --passages corpus.txt --out chat_sft_v4.jsonl

    # 2. fine-tune on the tree adapter (about 10 min: long passages)
    python -m memtree.finetune --model richardyoung/Phi-4-mini-instruct-heretic --adapter adapter_v3.pt --max-len 4096 \
        --synthetic 0 --data chat_sft_v4.jsonl --steps 400 --lr 1e-4 --rank 8 --out finetune_v4.pt

    # 3. the real recall test: original Phi vs tree Phi + memory tree
    python scripts/eval_recall_memory.py --adapter adapter_v3.pt --finetune finetune_v4.pt \
        --lengths 2048 4096 8192 --depths 0.1 0.5 0.9 --trials 5 | tee recall_memory.log

    # 4. chat
    python chat_mem.py   (edit it to load adapter_v3.pt + finetune_v4.pt)

The torch code for this mode has not run anywhere yet. If anything errors,
paste the traceback.
