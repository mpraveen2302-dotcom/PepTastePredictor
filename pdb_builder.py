# features/pdb_builder.py

def build_peptide_pdb(seq):
    lines = []
    x = 0.0
    for i, aa in enumerate(seq, 1):
        lines.append(
            f"ATOM  {i:5d}  CA  {aa} A{i:4d}    {x:8.3f} 0.000 0.000  1.00  0.00           C"
        )
        x += 3.8
    lines.append("END")
    return "\n".join(lines)
