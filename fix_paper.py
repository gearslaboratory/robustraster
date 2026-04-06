"""Fix paper_examples.ipynb function_tuning_config lines."""
import json, os

filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demos", "paper_examples.ipynb")

with open(filepath, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") != "code":
        continue
    src = cell.get("source", [])
    new_src = []
    for line in src:
        # Fix the bad \\n that should be \n in the function_tuning_config lines
        if "function_tuning_config={" in line and line.endswith("\\\\n\",\r\n") or line.endswith("\\\\n\",\n") or line.endswith('\\\\n"'):
            line = line.replace("\\\\n", "\\n")
        if '"tune_function": True' in line and line.endswith("\\\\n\",\r\n") or ('"tune_function": True' in line and line.endswith("\\\\n\",\n")):
            line = line.replace("\\\\n", "\\n")
        if line.strip().startswith('"    },') and "\\\\n" in line:
            line = line.replace("\\\\n", "\\n")
        new_src.append(line)
    cell["source"] = new_src

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')
print("[OK] Fixed paper_examples.ipynb")
