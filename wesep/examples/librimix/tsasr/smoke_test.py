from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/scratch/project_465002316/junyi/tse/ts_asr/wesep/examples/librimix/tsasr")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.test_ts_qwen3_asr import test_smoke


if __name__ == "__main__":
    test_smoke()
    print("one-batch smoke test passed")
