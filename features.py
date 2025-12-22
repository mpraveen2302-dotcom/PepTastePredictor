import numpy as np
from collections import Counter
from itertools import product
from Bio.SeqUtils.ProtParam import ProteinAnalysis

AA = "ACDEFGHIKLMNPQRSTVWY"

def clean_sequence(seq):
    if not isinstance(seq, str):
        return ""
    seq = seq.upper().replace(" ", "").replace("\n", "").replace("\t", "")
    return "".join(a for a in seq if a in AA)

def aa_composition(seq):
    c = Counter(seq)
    L = max(len(seq), 1)
    return {f"AA_{a}": c.get(a, 0)/L for a in AA}

def dipeptide_composition(seq):
    dipeptides = ["".join(p) for p in product(AA, repeat=2)]
    counts = Counter(seq[i:i+2] for i in range(len(seq)-1))
    L = max(len(seq)-1, 1)
    return {f"DP_{d}": counts.get(d, 0)/L for d in dipeptides}

def biophysical_features(seq):
    ana = ProteinAnalysis(seq)
    helix, turn, sheet = ana.secondary_structure_fraction()
    return {
        "length": len(seq),
        "molecular_weight": ana.molecular_weight(),
        "isoelectric_point": ana.isoelectric_point(),
        "aromaticity": ana.aromaticity(),
        "instability_index": ana.instability_index(),
        "gravy": ana.gravy(),
        "net_charge_pH7": ana.charge_at_pH(7.0),
        "helix_fraction": helix,
        "turn_fraction": turn,
        "sheet_fraction": sheet,
    }

def extract_features(seqs):
    rows = []
    for s in seqs:
        s = clean_sequence(s)
        f = {}
        f.update(biophysical_features(s))
        f.update(aa_composition(s))
        f.update(dipeptide_composition(s))
        rows.append(f)
    return rows
