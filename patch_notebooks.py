"""
All-in-one: fix paper_examples.ipynb, then patch the remaining notebooks.
Works by directly manipulating the JSON cell source arrays.
"""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
DEMOS = os.path.join(BASE, "demos")


# ===== 1. Fix paper_examples.ipynb (already partially patched with bad \\n) =====
paper_path = os.path.join(DEMOS, "paper_examples.ipynb")
with open(paper_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = cell["source"]
    # Find and fix the bad lines
    new_src = []
    for line in src:
        # These lines were injected with \\n (escaped) instead of \n
        if "function_tuning_config={" in line:
            line = '    function_tuning_config={\n'
        elif '"tune_function": True,' in line and "\\\\n" in repr(line):
            line = '        "tune_function": True,\n'
        # Fix },\\n -> },\n
        new_src.append(line)
    
    # Also fix the closing brace line: "    },\\n" -> "    },\n"
    final_src = []
    i = 0
    while i < len(new_src):
        line = new_src[i]
        # Check if this is the closing brace after tune_function
        if i > 0 and '"tune_function"' in new_src[i-1]:
            if line.rstrip("\r\n").endswith("\\n"):
                line = "    },\n"
        final_src.append(line)
        i += 1
    
    cell["source"] = final_src

with open(paper_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
print("[OK] Fixed paper_examples.ipynb")


# ===== 2. Patch the remaining notebooks =====
def patch_notebook(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        nb = json.load(f)

    changed = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue

        src = cell["source"]

        # Find a standalone tune_function line
        tune_idx = None
        tune_val = None
        for i, line in enumerate(src):
            text = line.strip().rstrip("\n")
            if text.startswith("tune_function") and "=" in text and "function_tuning_config" not in text and '"tune_function"' not in text:
                val = text.split("=", 1)[1].strip().rstrip(",")
                tune_val = val
                tune_idx = i
                break

        if tune_idx is None:
            continue

        changed = True

        # Remove the tune_function line
        new_src = [l for j, l in enumerate(src) if j != tune_idx]

        # Check if function_tuning_config exists
        ftc_idx = None
        for i, line in enumerate(new_src):
            if "function_tuning_config={" in line.replace(" ", ""):
                ftc_idx = i
                break

        if ftc_idx is not None:
            # Insert tune_function as first key right after the opening brace line
            new_src.insert(ftc_idx + 1, '        "tune_function": ' + tune_val + ',\n')
        else:
            # Insert a new function_tuning_config block before export_config
            insert_idx = None
            for i, line in enumerate(new_src):
                if "export_config" in line.replace(" ", ""):
                    insert_idx = i
                    break
            if insert_idx is not None:
                new_src.insert(insert_idx, '    function_tuning_config={\n')
                new_src.insert(insert_idx + 1, '        "tune_function": ' + tune_val + ',\n')
                new_src.insert(insert_idx + 2, '    },\n')

        cell["source"] = new_src

    if changed:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"[OK] Patched: {os.path.basename(filepath)}")
    else:
        print(f"[SKIP] No changes: {os.path.basename(filepath)}")


for name in ["ndvi_us_demo.ipynb", "ndvi_us_demo_r_example.ipynb", "demo_taylor_syn_treelist.ipynb"]:
    p = os.path.join(DEMOS, name)
    if os.path.exists(p):
        patch_notebook(p)
    else:
        print(f"[NOT FOUND] {name}")
