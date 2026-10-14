import argparse
import glob
import os
import shutil
import subprocess
import sys

MODEL_SHIM = '''"""NaVILA serving shim (installed by scripts/install_navila_sglang_shims.py).

sglang keys its model registry by the class name found in EntryClass, and our checkpoints carry
two different `architectures` spellings, so both are exported here and both resolve to the same
serving implementation.
"""
from agent_system.models.navila_sglang import LlavaLlamaModel


class NavilaForCausalLM(LlavaLlamaModel):
    """Architecture alias: the SFT checkpoints name the HF training class in config.json."""


EntryClass = [LlavaLlamaModel, NavilaForCausalLM]
'''

PROC_SHIM = '''"""NaVILA multimodal-processor shim (installed by scripts/install_navila_sglang_shims.py).

import_processors() keeps only classes whose __module__ is this module, so the processor is
subclassed here rather than imported, and it claims both architecture aliases.
"""
from agent_system.models.navila_processor import NaVILAMultimodalProcessor
from sglang.srt.models.navila import LlavaLlamaModel, NavilaForCausalLM


class NaVILAProcessor(NaVILAMultimodalProcessor):
    models = [LlavaLlamaModel, NavilaForCausalLM]
'''

CONFIG_PATCH = '''
# --- NaVILA (installed by scripts/install_navila_sglang_shims.py) -----------------------------
# get_config() below calls AutoConfig.from_pretrained, which raises on model_type "llava_llama"
# unless NaVILAConfig is registered first. sglang builds configs during its earliest bootstrap,
# before any repo code would otherwise import, so the registration happens here.
try:
    from agent_system.models.navila_config import register_navila_hf as _mpt_register_navila

    _mpt_register_navila()
except Exception:  # agent_system not on PYTHONPATH -> non-NaVILA use of this env, ignore
    pass
# --- end NaVILA --------------------------------------------------------------------------------
'''
PATCH_MARKER = "_mpt_register_navila"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, help="conda env prefix holding sglang")
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()

    sp = glob.glob(os.path.join(a.env, "lib", "python3.*", "site-packages"))
    if not sp:
        sys.exit("no site-packages under {}".format(a.env))
    srt = os.path.join(sp[0], "sglang", "srt")
    if not os.path.isdir(srt):
        sys.exit("sglang not installed in {}".format(a.env))

    model_shim = os.path.join(srt, "models", "navila.py")
    proc_shim = os.path.join(srt, "multimodal", "processors", "navila.py")
    hf_utils = os.path.join(srt, "utils", "hf_transformers_utils.py")

    if a.check_only:
        for p in (model_shim, proc_shim):
            print("{:<70s} {}".format(p, "OK" if os.path.exists(p) else "MISSING"))
        patched = PATCH_MARKER in open(hf_utils).read()
        print("{:<70s} {}".format(hf_utils, "PATCHED" if patched else "UNPATCHED"))
        return

    with open(model_shim, "w") as f:
        f.write(MODEL_SHIM)
    print("wrote", model_shim)
    with open(proc_shim, "w") as f:
        f.write(PROC_SHIM)
    print("wrote", proc_shim)

    src = open(hf_utils).read()
    if PATCH_MARKER in src:
        print("already patched", hf_utils)
    else:
        bak = hf_utils + ".before_navila_shim.bak"
        if not os.path.exists(bak):
            shutil.copy2(hf_utils, bak)
            print("backup ->", bak)
        # insert after the last sglang import at the top of the module, before any def/class
        lines = src.splitlines(keepends=True)
        anchor = 0
        for i, ln in enumerate(lines[:200]):
            if ln.startswith("from sglang.") or ln.startswith("import sglang"):
                anchor = i + 1
        if anchor == 0:
            sys.exit("could not find an sglang import anchor in {}".format(hf_utils))
        lines.insert(anchor, CONFIG_PATCH)
        open(hf_utils, "w").write("".join(lines))
        print("patched", hf_utils, "(after line {})".format(anchor))

    # --- verify ---
    py = os.path.join(a.env, "bin", "python")
    code = (
        "from sglang.srt.models.registry import ModelRegistry;"
        "from sglang.srt.managers.multimodal_processor import import_processors, PROCESSOR_MAPPING;"
        "import_processors('sglang.srt.multimodal.processors');"
        "archs=[k for k in ModelRegistry.models if 'avila' in k.lower() or 'Navila' in k];"
        "print('model archs:', archs);"
        "print('processor archs:', [c.__name__ for c in PROCESSOR_MAPPING "
        "if 'avila' in c.__name__.lower()]);"
        "from transformers import AutoConfig;"
        "import agent_system.models.navila_config as nc;"
        "print('llava_llama registered:', 'llava_llama' in AutoConfig._model_mapping._model_mapping "
        "or 'llava_llama' in str(AutoConfig._model_mapping.keys()))"
    )
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["LD_LIBRARY_PATH"] = os.path.join(a.env, "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    print("--- verify ---", flush=True)
    subprocess.call([py, "-c", code], env=env)


if __name__ == "__main__":
    main()
