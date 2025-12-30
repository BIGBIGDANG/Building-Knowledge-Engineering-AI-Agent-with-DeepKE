import hydra
import os
import torch
import logging
from hydra import utils
import json
import csv
import re
from typing import List, Dict, Tuple, Any
from pyld import jsonld

from preprocess import _handle_relation_data, _lm_serialize
from utils import load_csv
from LMModel import LM
from InferBert import InferNer

logger = logging.getLogger(__name__)


# ---------- JSON-LD helpers (skip null/invalid IRI) ----------
def normalize_url(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    if s.lower() in ("null", "none", "nil", "nan", ""):
        return ""
    return s


def get_jsonld_compacted(head: str, rel: str, tail: str, url: str) -> Dict[str, Any]:
    """
    直接跳过 url 为 null/空 的关系：调用方负责 url 为空时不调用本函数。
    这里仍保留 try/except，防止 url 格式虽非空但不合法导致程序崩。
    """
    doc = {
        "@id": head,
        url: {"@id": tail},
    }
    context = {rel: url}
    try:
        return jsonld.compact(doc, context)
    except Exception as e:
        logger.warning(f"JSON-LD compact failed, skip. rel={rel}, url={url}, err={e}")
        return {}


# ---------- data / preprocessing ----------
def init_rels(cfg):
    relation_data = load_csv(os.path.join(cfg.cwd, cfg.data_path, "relation.csv"), verbose=False)
    rels = _handle_relation_data(relation_data)
    return rels


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"([。！？；;.!?])", text)
    sents = []
    buf = ""
    for p in parts:
        if not p:
            continue
        buf += p
        if re.match(r"[。！？；;.!?]", p):
            sents.append(buf.strip())
            buf = ""
    if buf.strip():
        sents.append(buf.strip())
    return [s for s in sents if s]


def chunk_text(text: str, chunk_max_len: int = 300, overlap: int = 50) -> List[str]:
    sents = split_sentences(text.strip())
    if not sents:
        return []

    chunks = []
    cur = ""
    for s in sents:
        if len(cur) + len(s) <= chunk_max_len:
            cur += s
        else:
            if cur:
                chunks.append(cur)
                tail = cur[-overlap:] if overlap > 0 and len(cur) > overlap else cur
                cur = tail + s
            else:
                chunks.append(s[:chunk_max_len])
                rest = s[chunk_max_len:]
                while rest:
                    prev = chunks[-1]
                    tail = prev[-overlap:] if overlap > 0 and len(prev) > overlap else prev
                    piece = rest[:chunk_max_len]
                    chunks.append(tail + piece)
                    rest = rest[chunk_max_len:]
                cur = ""
    if cur:
        chunks.append(cur)
    return [c.strip() for c in chunks if c.strip()]


def extract_entities(ner_result: List[Tuple[str, str]], label2word: Dict[str, str]) -> List[Tuple[str, str]]:
    entities: List[Tuple[str, str]] = []
    temp = ""
    last_type = None

    def flush():
        nonlocal temp, last_type
        if temp and last_type in label2word:
            entities.append((temp, label2word[last_type]))
        temp = ""
        last_type = None

    for tok, tag in ner_result:
        if not tag or tag == "O":
            flush()
            continue

        if tag.startswith("B-"):
            flush()
            last_type = tag[2:]
            temp = tok
        elif tag.startswith("I-"):
            t = tag[2:]
            if last_type == t and temp:
                temp += tok
            else:
                flush()
                last_type = t
                temp = tok
        else:
            flush()

    flush()
    return entities


def build_instances_for_chunk(text_chunk: str, entities: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    instances: List[Dict[str, str]] = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            head, head_type = entities[i]
            tail, tail_type = entities[j]
            instances.append(
                {
                    "sentence": text_chunk.strip(),
                    "head": head.strip(),
                    "tail": tail.strip(),
                    "head_type": head_type.strip(),
                    "tail_type": tail_type.strip(),
                }
            )
    return instances


def pad_batch_token2idx(instances: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
    lens = [int(ins["seq_len"]) for ins in instances]
    max_len = max(lens) if lens else 0
    if max_len == 0:
        return torch.empty(0, 0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    batch = []
    for ins in instances:
        ids = ins["token2idx"]
        ids = ids[:max_len] + [0] * max(0, max_len - len(ids))
        batch.append(ids)

    word = torch.tensor(batch, dtype=torch.long)
    lens_t = torch.tensor(lens, dtype=torch.long)
    return word, lens_t


def write_csv(rows: List[Dict[str, Any]], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    headers = [
        "chunk_id",
        "chunk_text",
        "head",
        "head_type",
        "relation",
        "relation_url",
        "tail",
        "tail_type",
        "confidence",
        "jsonld",
    ]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    cwd = utils.get_original_cwd()
    cfg.cwd = cwd

    # ---- label2word / rel2url ----
    label2word = {}
    with open(os.path.join(cfg.cwd, cfg.data_path, "type.txt"), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            w, lab = line.split(" ", 1)
            label2word[lab.strip()] = w.strip()
    logger.info(label2word)

    rel2url = {}
    with open(os.path.join(cfg.cwd, cfg.data_path, "url.txt"), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            k, v = line.split(" ", 1)
            rel2url[k.strip()] = normalize_url(v.strip())  # 'null' -> ''
    logger.info(rel2url)

    # ---- init once ----
    ner_model = InferNer(cfg.nerfp)
    rels = init_rels(cfg)

    device = torch.device("cpu")
    re_model = LM(cfg)
    re_model.load(cfg.refp, device=device)
    re_model.to(device)
    re_model.eval()

    # ---- configs ----
    text = cfg.text
    chunk_max_len = int(getattr(cfg, "chunk_max_len", 300))
    chunk_overlap = int(getattr(cfg, "chunk_overlap", 50))
    batch_size = int(getattr(cfg, "batch_size", 32))
    out_csv = getattr(cfg, "out_csv", os.path.join(cfg.cwd, "outputs", "triples.csv"))
    conf_th = float(getattr(cfg, "confidence_threshold", 0.0))

    chunks = chunk_text(text, chunk_max_len=chunk_max_len, overlap=chunk_overlap)
    logger.info(f"Total chunks: {len(chunks)}")

    rel_list = list(rels.keys())
    all_rows: List[Dict[str, Any]] = []

    for chunk_id, chunk in enumerate(chunks):
        ner_res = ner_model.predict(chunk)
        entities = extract_entities(ner_res, label2word)
        if len(entities) < 2:
            continue

        instances = build_instances_for_chunk(chunk, entities)
        if not instances:
            continue

        _lm_serialize(instances, cfg)

        for start in range(0, len(instances), batch_size):
            batch_insts = instances[start : start + batch_size]

            word, lens_t = pad_batch_token2idx(batch_insts)
            if word.numel() == 0:
                continue

            # FIX: sort lens desc for pack_padded_sequence(enforce_sorted=True)
            lens_sorted, perm_idx = torch.sort(lens_t, descending=True)
            word_sorted = word.index_select(0, perm_idx)

            x = {"word": word_sorted.to(device), "lens": lens_sorted.to(device)}
            with torch.no_grad():
                logits = re_model(x)
                probs = torch.softmax(logits, dim=-1)
                max_prob, max_idx = probs.max(dim=-1)

            inv_perm = torch.empty_like(perm_idx)
            inv_perm[perm_idx] = torch.arange(len(perm_idx))
            max_prob = max_prob.index_select(0, inv_perm)
            max_idx = max_idx.index_select(0, inv_perm)

            for b, ins in enumerate(batch_insts):
                prob = float(max_prob[b].item())
                rel_name = rel_list[int(max_idx[b].item())]
                if prob < conf_th:
                    continue

                rel_url = rel2url.get(rel_name, "")
                # 关键：url 为 null/空 直接跳过 JSON-LD
                compacted = {}
                if rel_url:
                    compacted = get_jsonld_compacted(ins["head"], rel_name, ins["tail"], rel_url)

                row = {
                    "chunk_id": chunk_id,
                    "chunk_text": chunk,
                    "head": ins["head"],
                    "head_type": ins["head_type"],
                    "relation": rel_name,
                    "relation_url": rel_url,
                    "tail": ins["tail"],
                    "tail_type": ins["tail_type"],
                    "confidence": f"{prob:.4f}",
                    "jsonld": json.dumps(compacted, ensure_ascii=False),
                }
                all_rows.append(row)

                logger.info(f"\"{ins['head']}\" 和 \"{ins['tail']}\" 关系：\"{rel_name}\"，置信度 {prob:.2f}")
                if compacted:
                    logger.info(json.dumps(compacted, indent=2, ensure_ascii=False))

    write_csv(all_rows, out_csv)
    logger.info(f"Saved {len(all_rows)} triples to: {out_csv}")


if __name__ == "__main__":
    main()
