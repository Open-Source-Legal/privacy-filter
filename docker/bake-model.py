"""Pre-download HF model weights into the image's HF_HOME.

Invoked by the Dockerfile builder stage so that the runtime image is
self-contained: cold starts and air-gapped deployments do not need outbound
network. Reads MODEL_ID and (optionally) MODEL_REVISION from the environment.
"""

from __future__ import annotations

import os
import sys

from transformers import AutoModelForTokenClassification, AutoTokenizer


def main() -> int:
    model_id = os.environ.get("MODEL_ID", "openai/privacy-filter")
    revision = os.environ.get("MODEL_REVISION") or None
    kwargs = {"revision": revision} if revision else {}

    AutoTokenizer.from_pretrained(model_id, **kwargs)
    AutoModelForTokenClassification.from_pretrained(model_id, **kwargs)

    hf_home = os.environ.get("HF_HOME", "(unset)")
    print(f"baked {model_id}@{revision or 'default'} into HF_HOME={hf_home}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
