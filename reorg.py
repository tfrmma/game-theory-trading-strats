"""
Reorganizes the repo into engine/ strategies/ backtesting/ and fixes every
internal import to match. Run from the repo root:

    python reorg.py

Works identically in PowerShell, cmd, or bash - it's pure Python, no shell
scripting involved. Uses `git mv` under the hood so file history is preserved.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

ENGINE_MODS = ["init", "central_risk_manager", "runner", "hyperliquid_feed", "hot_paths"]
STRAT_MODS = [
    "spoofing_counter", "predatory_liquidity", "queue_warfare", "funding_arbitrage",
    "liquidation_frontrun", "adaptive_guerrilla", "adverse_selection", "info_asymmetry",
]
BT_MODS = ["tick_by_tick_backtester", "RL_tuner"]

MOVES = (
    [(m + ".py", f"engine/{m}.py") for m in ENGINE_MODS]
    + [("_hot_paths.pyx", "engine/_hot_paths.pyx"), ("_hot_paths_pure.py", "engine/_hot_paths_pure.py"),
       ("setup_hot_paths.py", "engine/setup_hot_paths.py")]
    + [(m + ".py", f"strategies/{m}.py") for m in STRAT_MODS]
    + [(m + ".py", f"backtesting/{m}.py") for m in BT_MODS]
)


def run(cmd):
    print(">", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print("  FAILED:", result.stderr.strip())
        sys.exit(1)


def step1_move_files():
    for folder in ("engine", "strategies", "backtesting"):
        (ROOT / folder).mkdir(exist_ok=True)
        init_file = ROOT / folder / "__init__.py"
        if not init_file.exists():
            init_file.write_text("")
        run(["git", "add", f"{folder}/__init__.py"])

    for src, dst in MOVES:
        if not (ROOT / src).exists():
            print(f"  SKIP (already moved or missing): {src}")
            continue
        run(["git", "mv", src, dst])


def step2_fix_imports():
    all_py = list((ROOT / "engine").rglob("*.py")) + \
             list((ROOT / "strategies").rglob("*.py")) + \
             list((ROOT / "backtesting").rglob("*.py")) + \
             list((ROOT / "tests").rglob("*.py"))

    for f in all_py:
        text = f.read_text(encoding="utf-8")
        original = text

        for m in ENGINE_MODS:
            text = re.sub(rf"^from {m} import", f"from engine.{m} import", text, flags=re.M)
            text = re.sub(rf"^([ \t]+)from {m} import", rf"\1from engine.{m} import", text, flags=re.M)
            text = re.sub(rf"^import {m}$", f"import engine.{m} as {m}", text, flags=re.M)
            text = re.sub(rf"^([ \t]+)import {m}$", rf"\1import engine.{m} as {m}", text, flags=re.M)
            text = text.replace(f'"{m}.Info"', f'"engine.{m}.Info"')

        for m in STRAT_MODS:
            text = re.sub(rf"^from {m} import", f"from strategies.{m} import", text, flags=re.M)
            text = re.sub(rf"^([ \t]+)from {m} import", rf"\1from strategies.{m} import", text, flags=re.M)

        for m in BT_MODS:
            text = re.sub(rf"^from {m} import", f"from backtesting.{m} import", text, flags=re.M)
            text = re.sub(rf"^([ \t]+)from {m} import", rf"\1from backtesting.{m} import", text, flags=re.M)
            text = re.sub(rf"^([ \t]+)import {m}$", rf"\1import backtesting.{m} as {m}", text, flags=re.M)

        if text != original:
            f.write_text(text, encoding="utf-8")
            print(f"  fixed imports: {f.relative_to(ROOT)}")


def step3_one_off_fixes():
    # hot_paths.py imports its own Cython/pure-Python siblings
    hp = ROOT / "engine" / "hot_paths.py"
    text = hp.read_text(encoding="utf-8")
    text = text.replace("from _hot_paths import (", "from engine._hot_paths import (")
    text = text.replace("from _hot_paths_pure import (", "from engine._hot_paths_pure import (")
    hp.write_text(text, encoding="utf-8")

    # aliased imports the generic regex above doesn't touch (alias != module name)
    tri = ROOT / "tests" / "test_runner_integration.py"
    text = tri.read_text(encoding="utf-8")
    text = text.replace("import runner as runner_mod", "import engine.runner as runner_mod")
    text = re.sub(r"^([ \t]+)import RL_tuner$", r"\1import backtesting.RL_tuner as RL_tuner", text, flags=re.M)
    tri.write_text(text, encoding="utf-8")

    # stale internal print() string left over from before this session's very first fix
    rl = ROOT / "backtesting" / "RL_tuner.py"
    text = rl.read_text(encoding="utf-8")
    text = text.replace(
        "from rl_param_tuner import train_agent, TrainConfig",
        "from backtesting.RL_tuner import train_agent, TrainConfig",
    )
    rl.write_text(text, encoding="utf-8")

    print("  one-off fixes applied")


def step4_fix_pyproject():
    pp = ROOT / "pyproject.toml"
    text = pp.read_text(encoding="utf-8")
    text = text.replace('runner = "runner:main"', 'runner = "engine.runner:main"')
    text = re.sub(r'^where = \["\."\]\n', "", text, flags=re.M)
    text = text.replace('include = ["*"]', 'include = ["engine*", "strategies*", "backtesting*"]')
    pp.write_text(text, encoding="utf-8")
    print("  pyproject.toml updated")


if __name__ == "__main__":
    print("=== step 1: moving files (git mv) ===")
    step1_move_files()
    print("=== step 2: fixing imports ===")
    step2_fix_imports()
    print("=== step 3: one-off fixes ===")
    step3_one_off_fixes()
    print("=== step 4: pyproject.toml ===")
    step4_fix_pyproject()
    print("\nDone. Now run:")
    print("  pip install -e \".[dev]\"")
    print("  pytest -v")
