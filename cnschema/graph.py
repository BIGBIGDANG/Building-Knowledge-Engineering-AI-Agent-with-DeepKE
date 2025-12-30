import os
import argparse
import pandas as pd
import networkx as nx

def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    将 (head, relation, tail, confidence) 构成有向语义网络。
    同一对(head, tail)如果出现多个relation，保留置信度最高的一条（也可改为合并）。
    """
    G = nx.DiGraph()

    # 用于去重：对同一 (head, tail, relation) 取最大置信度
    df = df.groupby(["head", "tail", "relation"], as_index=False)["confidence"].max()

    # 如果同一 (head, tail) 存在多个 relation，取置信度最高的那条
    df = df.sort_values("confidence", ascending=False)
    df = df.drop_duplicates(subset=["head", "tail"], keep="first")

    for _, row in df.iterrows():
        h = str(row["head"]).strip()
        t = str(row["tail"]).strip()
        r = str(row["relation"]).strip()
        c = float(row["confidence"])

        if not h or not t or not r:
            continue

        if not G.has_node(h):
            G.add_node(h)
        if not G.has_node(t):
            G.add_node(t)

        G.add_edge(h, t, label=r, confidence=c)

    return G


def save_html_pyvis(G: nx.DiGraph, out_html: str):
    try:
        from pyvis.network import Network
    except ImportError:
        raise RuntimeError("缺少 pyvis：请先 pip install pyvis")

    net = Network(height="800px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222222")
    net.barnes_hut()

    for n in G.nodes():
        net.add_node(n, label=n, title=n)

    for u, v, data in G.edges(data=True):
        label = data.get("label", "")
        conf = data.get("confidence", 0.0)
        title = f"{u} --{label}--> {v}\nconfidence={conf:.4f}"
        # edge label 只显示关系名；把置信度放到 tooltip 里
        net.add_edge(u, v, label=label, title=title, value=max(conf, 0.01))

    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    net.write_html(out_html)


def save_png_matplotlib(G: nx.DiGraph, out_png: str):
    import matplotlib.pyplot as plt

    # 中文字体（有就用，没有也不强求）
    try:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    if G.number_of_nodes() == 0:
        raise RuntimeError("图为空：没有满足阈值的三元组。")

    # 布局：节点多时 spring_layout 会慢一点，可改为 kamada_kawai_layout
    pos = nx.spring_layout(G, seed=42, k=0.5)

    plt.figure(figsize=(20, 10))
    nx.draw_networkx_nodes(G, pos, node_size=2000)
    nx.draw_networkx_labels(G, pos, font_size=10)

    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", arrowsize=18, width=1.4)

    edge_labels = {(u, v): d.get("label", "") for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

    plt.axis("off")
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=str, default="outputs/triples.csv", help="triple.csv 路径")
    ap.add_argument("--min_conf", type=float, default=0.8, help="置信度阈值")
    ap.add_argument("--out_html", type=str, default="outputs/kg.html", help="输出交互式HTML")
    ap.add_argument("--out_png", type=str, default="", help="输出静态PNG（留空则不输出）")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv, encoding="utf-8-sig")

    required = {"head", "tail", "relation", "confidence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要列：{missing}，现有列：{list(df.columns)}")

    # confidence 可能是字符串，强制转 float
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df.dropna(subset=["confidence", "head", "tail", "relation"])

    df_f = df[df["confidence"] > args.min_conf].copy()

    print(f"[INFO] total rows={len(df)}, kept(conf>{args.min_conf})={len(df_f)}")

    G = build_graph(df_f)

    print(f"[INFO] graph nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    # 交互式 HTML（推荐）
    save_html_pyvis(G, args.out_html)
    print(f"[OK] saved html: {args.out_html}")

    # 可选：静态 PNG
    if args.out_png:
        save_png_matplotlib(G, args.out_png)
        print(f"[OK] saved png: {args.out_png}")


if __name__ == "__main__":
    main()
