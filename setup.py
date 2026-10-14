import os
from pathlib import Path

from setuptools import find_namespace_packages, setup

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, "src/verl/version/version")) as f:
    __version__ = f.read().strip()

install_requires = [
    "accelerate",
    "codetiming",
    "datasets",
    "litellm",
    "json5",
    "tiktoken",
    "requests",
    "dill",
    "hydra-core",
    "numpy",
    "pandas",
    "peft",
    "pyarrow>=19.0.0",
    "pybind11",
    "ray[default]>=2.41.0",
    "torchdata",
    "tensordict",
    "transformers>=4.52.1",
    "wandb",
    "packaging>=20.0",
    "qwen-vl-utils[decord]",
]

TEST_REQUIRES = ["pytest", "pre-commit", "py-spy"]
GPU_REQUIRES = ["liger-kernel"]  # flash-attn is installed separately (a prebuilt wheel matching your torch/CUDA)
# Trainer env A (Minecraft + NaVILA): must match requirements_sglang.txt.
SGLANG_REQUIRES = [
    "tensordict",
    "sglang[srt,openai]==0.5.2",
    "torch-memory-saver>=0.0.5",
    "torch==2.8.0",
]

extras_require = {
    "test": TEST_REQUIRES,
    "gpu": GPU_REQUIRES,
    "sglang": SGLANG_REQUIRES,
}


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="mpt",
    version=__version__,
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src", include=["verl*", "agent_system*", "economy_engine*"]),
    license="Apache 2.0",
    description="MPT: sealed-bid auction societies of LLM agents trained with revenue-share settlement "
                "and alternating act/counterfactual-bid RL (built on a vendored verl fork)",
    install_requires=install_requires,
    extras_require=extras_require,
    package_data={
        "": ["version/*"],
        "verl": ["trainer/config/*.yaml", "trainer/runtime_env.yaml", "version/version"],
    },
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
)