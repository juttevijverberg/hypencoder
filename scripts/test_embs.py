import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from hypencoder_cb.modeling.hypencoder_bebase import TextDualEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HF_MODEL = "facebook/contriever-msmarco"
CHECKPOINT = "/scratch-shared/scur1744/models/contriever_freeze_norm/checkpoint-11790"

TEXT = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."
)

def mean_pool(last_hidden, mask):
    mask = mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

# -----------------------------
# 1) HF RetroMAE (backbone)
# -----------------------------
tok_hf = AutoTokenizer.from_pretrained(HF_MODEL)
model_hf = AutoModel.from_pretrained(HF_MODEL).to(DEVICE).eval()

inp_hf = tok_hf(TEXT, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
with torch.no_grad():
    out_hf = model_hf(**inp_hf)

# If your pooling is CLS, use CLS here (not mean_pool)
emb_hf = out_hf.last_hidden_state
mask = inp_hf["attention_mask"].unsqueeze(-1).type_as(emb_hf)
emb_hf = (emb_hf * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

# -----------------------------
# 2) Checkpoint TextDualEncoder (wrapper output)
# -----------------------------
model_ckpt = TextDualEncoder.from_pretrained(CHECKPOINT).to(DEVICE).eval()
tok_ckpt = AutoTokenizer.from_pretrained(CHECKPOINT)

inp_ck = tok_ckpt(TEXT, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
with torch.no_grad():
    out_ck_wrapper = model_ckpt.passage_encoder(
        input_ids=inp_ck["input_ids"],
        attention_mask=inp_ck["attention_mask"],
    )

emb_ckpt = F.normalize(out_ck_wrapper.representation, dim=-1)

# -----------------------------
# Tokenization check
# -----------------------------
print("HF tokenizer name_or_path:", tok_hf.name_or_path)
print("CKPT tokenizer name_or_path:", tok_ckpt.name_or_path)

ids_hf = tok_hf(TEXT, return_tensors="pt", truncation=True, max_length=128)["input_ids"][0]
ids_ck = tok_ckpt(TEXT, return_tensors="pt", truncation=True, max_length=128)["input_ids"][0]
print("Same input_ids?", torch.equal(ids_hf, ids_ck))
print("HF ids head:", ids_hf[:20].tolist())
print("CK ids head:", ids_ck[:20].tolist())

# -----------------------------
# Wrapper embedding comparison
# -----------------------------
cos_wrapper = F.cosine_similarity(emb_hf, emb_ckpt).item()
print("\n=== Wrapper representation comparison ===")
print("HF embedding dim:", emb_hf.shape[-1])
print("CKPT embedding dim:", emb_ckpt.shape[-1])
print("HF norm:", emb_hf.norm().item())
print("CKPT norm:", emb_ckpt.norm().item())
print("Cosine similarity (HF CLS vs CKPT passage_encoder.representation):", cos_wrapper)

# ============================================================
# Backbone-level comparison (CLS + meanpool)
# ============================================================
print("\n=== Backbone-level comparison (transformer outputs) ===")

# Use ONE tokenizer + ONE set of ids to avoid any subtle differences
inp_shared = tok_hf(TEXT, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)

with torch.no_grad():
    hf_last = model_hf(**inp_shared).last_hidden_state  # [1,L,768]

    pe = model_ckpt.passage_encoder
    print("CKPT passage_encoder pooling_type:", getattr(pe, "pooling_type", None))

    # Try to locate the HF backbone module inside the checkpoint passage encoder
    backbone = None
    for attr in ["transformer", "encoder", "model"]:
        if hasattr(pe, attr):
            backbone = getattr(pe, attr)
            break

    if backbone is None:
        attrs = [a for a in dir(pe) if not a.startswith("_")]
        print("Could not find backbone via ['transformer','encoder','model'] on passage_encoder.")
        print("passage_encoder attrs:", attrs)
        raise RuntimeError("Backbone not found. Tell me what attribute holds the HF model.")

    # Optional: check for a pooler on the HF backbone (BERT-like models sometimes have it)
    print("Backbone class:", type(backbone).__name__)
    print("Backbone has pooler?:", hasattr(backbone, "pooler"))

    ck_last = backbone(
        input_ids=inp_shared["input_ids"],
        attention_mask=inp_shared["attention_mask"],
    ).last_hidden_state

# Compare CLS token vectors (HF backbone vs CKPT backbone)
cls_hf = F.normalize(hf_last[:, 0], dim=-1)
cls_ck = F.normalize(ck_last[:, 0], dim=-1)
cos_cls = F.cosine_similarity(cls_hf, cls_ck).item()
print("CLS cosine (HF backbone vs CKPT backbone):", cos_cls)

# Compare masked mean pooled backbone vectors
mp_hf = F.normalize(mean_pool(hf_last, inp_shared["attention_mask"]), dim=-1)
mp_ck = F.normalize(mean_pool(ck_last, inp_shared["attention_mask"]), dim=-1)
cos_backbone_mean = F.cosine_similarity(mp_hf, mp_ck).item()
print("Meanpool cosine (HF backbone vs CKPT backbone):", cos_backbone_mean)

# ============================================================
# NEW: Is CKPT wrapper rep actually the CKPT backbone CLS?
# ============================================================
with torch.no_grad():
    out_ck_wrapper_shared = pe(
        input_ids=inp_shared["input_ids"],
        attention_mask=inp_shared["attention_mask"],
    )
    rep_ck = out_ck_wrapper_shared.representation
    rep_ck_n = F.normalize(rep_ck, dim=-1)

    ck_cls_n = F.normalize(ck_last[:, 0], dim=-1)

cos_rep_vs_cls = F.cosine_similarity(rep_ck_n, ck_cls_n).item()
print("\n=== CKPT internal consistency check ===")
print("Cosine(rep, CKPT backbone CLS):", cos_rep_vs_cls)

emb_hf = F.normalize(hf_last[:, 0], dim=-1)
emb_ckpt = F.normalize(out_ck_wrapper_shared.representation, dim=-1)

print("HF norm:", emb_hf.norm().item())
print("CKPT norm:", emb_ckpt.norm().item())
print("Cosine(HF CLS, CKPT rep):", F.cosine_similarity(emb_hf, emb_ckpt).item())