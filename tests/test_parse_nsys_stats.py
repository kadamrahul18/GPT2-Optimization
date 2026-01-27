import importlib.util
from pathlib import Path


def load_parser_module():
    repo_root = Path(__file__).resolve().parents[1]
    parser_path = repo_root / "scripts" / "profiling" / "parse_nsys_stats.py"
    spec = importlib.util.spec_from_file_location("parse_nsys_stats", parser_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_ranked_items_parses_nsys_tables():
    module = load_parser_module()
    text = """
 ** NVTX Range Summary (nvtx_sum):

 Time (%)   Total Time (ns)    Instances     Avg (ns)        Med (ns)       Min (ns)       Max (ns)      StdDev (ns)     Style                  Range
 --------  ------------------  ---------  --------------  --------------  ------------  --------------  --------------  -------  ------------------------------------
     53.9  27,65,96,75,62,725     14,244  19,41,84,748.9  12,36,84,299.5  11,37,90,285  1,93,66,43,699  11,76,77,819.7  PushPop  :DeepSpeedEngine.backward
     40.8  20,93,35,96,90,089     14,244  14,69,64,314.1   7,52,43,705.0   6,79,65,697  1,84,03,11,805  11,73,92,154.2  PushPop  :DeepSpeedEngine.allreduce_gradients

 ** OS Runtime Summary (osrt_sum):

 Time (%)    Total Time (ns)      Num Calls       Avg (ns)           Med (ns)       Min (ns)         Max (ns)           StdDev (ns)               Name
 --------  --------------------  -----------  -----------------  ----------------  -----------  ------------------  -------------------  ----------------------
     21.9  5,62,65,05,87,29,882     3,46,812     16,22,35,040.1           2,857.0        1,493      5,00,50,61,007       29,81,22,921.2  poll
     21.0  5,38,64,74,73,42,078     2,77,538     19,40,80,620.8     2,35,96,440.5       89,826  15,39,92,94,62,355    13,37,35,03,709.2  pthread_cond_wait
    """
    nvtx = module.extract_ranked_items(text, kind="nvtx")
    osrt = module.extract_ranked_items(text, kind="osrt")

    assert nvtx
    assert osrt
    assert nvtx[0]["name"].endswith("DeepSpeedEngine.backward")
    assert "allreduce_gradients" in nvtx[1]["name"]
    assert osrt[0]["name"] == "poll"
    assert osrt[1]["name"] == "pthread_cond_wait"
