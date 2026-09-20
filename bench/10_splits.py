"""Component-level train/val/test split — the only split that is not leaky.

WHY NOT pdb_id. The corpus was selected by greedy set-cover over 40%
sequence-identity clusters (modal_pdb.py): an entry is kept only if it covers a
NEW cluster. That deduplicates the ARCHIVE, but a kept entry still contains
chains from clusters already covered, so kept entries remain homologous to each
other. Measured on this corpus: 12,349 of 38,833 entries (31.8%) share a 40%
cluster with another entry. A split on pdb_id therefore leaks homologues across
the boundary for nearly a third of the corpus.

So: build the bipartite entry<->cluster graph, take connected components, and
split on COMPONENTS. Two entries in different components share no 40% cluster,
by construction. The audit at the end asserts exactly that.

au and asm1 of one entry are two objects of the SAME structure, so grouping on
pdb_id (which the component carries) keeps them on the same side automatically.
"""
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

CLUSTER_URL = ("https://cdn.rcsb.org/resources/sequence/clusters/"
               "clusters-by-entity-40.txt")
CLUSTER_FILE = f"{C.AUX}/clusters-by-entity-40.txt"
FRACTIONS = {"train": 0.80, "val": 0.10, "test": 0.10}
SEED = 20260919


def ensure_clusters():
    os.makedirs(C.AUX, exist_ok=True)
    if os.path.exists(CLUSTER_FILE) and os.path.getsize(CLUSTER_FILE) > 1_000_000:
        print(f"cluster file present: {CLUSTER_FILE}")
        return
    import urllib.request
    print(f"downloading {CLUSTER_URL}")
    tmp = CLUSTER_FILE + ".tmp"
    urllib.request.urlretrieve(CLUSTER_URL, tmp)
    os.replace(tmp, CLUSTER_FILE)
    print(f"  wrote {os.path.getsize(CLUSTER_FILE)/1e6:.0f} MB")


def main():
    ensure_clusters()
    man = C.load_manifest(columns=["pdb_id", "object_id"])
    entries = sorted(set(man.pdb_id.str.lower()))
    entset = set(entries)
    print(f"rendered entries : {len(entries):,}")
    print(f"rendered objects : {man.object_id.nunique():,}")

    # bipartite entry <-> cluster, restricted to entries we actually rendered
    cl2ent = defaultdict(list)
    n_clusters = 0
    with open(CLUSTER_FILE) as fh:
        for ci, line in enumerate(fh):
            n_clusters += 1
            for tok in line.split():
                e = tok.split("_")[0].lower()
                if e in entset:
                    cl2ent[ci].append(e)
    print(f"40% clusters total          : {n_clusters:,}")
    print(f"clusters touching the corpus: {len(cl2ent):,}")

    # union-find over entries sharing any cluster
    parent = {e: e for e in entries}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    shared = 0
    for members in cl2ent.values():
        uniq = set(members)
        if len(uniq) > 1:
            shared += 1
            it = iter(uniq)
            first = next(it)
            for other in it:
                union(first, other)
    print(f"clusters hit by >1 entry    : {shared:,}")

    comp = defaultdict(list)
    for e in entries:
        comp[find(e)].append(e)
    comps = sorted(comp.values(), key=len, reverse=True)
    sizes = np.array([len(c) for c in comps])
    print(f"connected components        : {len(comps):,}")
    print(f"  singletons {int((sizes == 1).sum()):,} | "
          f"largest {sizes[0]:,} ({100*sizes[0]/len(entries):.2f}% of corpus)")

    # Greedy largest-first bin packing toward the target fractions. Largest-first
    # matters: the biggest component is 7.8% of the corpus, so a random
    # assignment could hand the whole of it to a 10% test split and blow the
    # proportions. It is deterministic given SEED.
    rng = np.random.default_rng(SEED)
    names = list(FRACTIONS)
    target = {k: FRACTIONS[k] * len(entries) for k in names}
    got = {k: 0 for k in names}
    assign = {}
    order = sorted(range(len(comps)), key=lambda i: (-len(comps[i]),
                                                     comps[i][0]))
    for i in order:
        # most under-quota split wins; jitter breaks ties reproducibly
        deficit = {k: (target[k] - got[k]) / max(target[k], 1) for k in names}
        jit = rng.random(len(names)) * 1e-9
        pick = names[int(np.argmax([deficit[k] + jit[j]
                                    for j, k in enumerate(names)]))]
        for e in comps[i]:
            assign[e] = pick
        got[pick] += len(comps[i])
    print("\nentry counts by split:")
    for k in names:
        print(f"  {k:5s} {got[k]:7,}  ({100*got[k]/len(entries):5.2f}%, "
              f"target {100*FRACTIONS[k]:.0f}%)")

    comp_id = {}
    for i, c in enumerate(comps):
        for e in c:
            comp_id[e] = i
    df = pd.DataFrame({
        "pdb_id": entries,
        "component": [comp_id[e] for e in entries],
        "component_size": [len(comps[comp_id[e]]) for e in entries],
        "split": [assign[e] for e in entries],
    })

    # ---- AUDIT: the property the whole split exists to guarantee -----------
    ent2split = dict(zip(df.pdb_id, df.split))
    bad = 0
    for members in cl2ent.values():
        s = {ent2split[e] for e in set(members)}
        if len(s) > 1:
            bad += 1
    print(f"\nAUDIT clusters straddling a split boundary: {bad}")
    if bad != 0:
        raise SystemExit("LEAK: a 40% cluster spans two splits - refusing to write")

    obj = man.drop_duplicates("object_id")[["object_id", "pdb_id"]].copy()
    obj["pdb_id_l"] = obj.pdb_id.str.lower()
    obj = obj.merge(df, left_on="pdb_id_l", right_on="pdb_id",
                    suffixes=("", "_y"))
    print("object counts by split:")
    print("  " + str(obj.split.value_counts().to_dict()))

    os.makedirs(C.META, exist_ok=True)
    tmp = C.SPLITS + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, C.SPLITS)
    print(f"\nwrote {C.SPLITS}  ({len(df):,} entries)")

    C.atomic_write_json(f"{C.RESULTS}/splits_summary.json", {
        "seed": SEED, "n_entries": len(entries),
        "n_objects": int(man.object_id.nunique()),
        "n_components": len(comps),
        "largest_component": int(sizes[0]),
        "n_singleton_components": int((sizes == 1).sum()),
        "clusters_total": n_clusters,
        "clusters_touching_corpus": len(cl2ent),
        "clusters_multi_entry": shared,
        "entries_by_split": got,
        "objects_by_split": obj.split.value_counts().to_dict(),
        "audit_clusters_straddling_splits": bad,
    })


if __name__ == "__main__":
    main()
