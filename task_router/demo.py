"""
SatQuery AI Task Router Demo.
Tests all official SIH representative queries and prints the structured results.
"""

import sys
from pathlib import Path

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task_router.inference.router_engine import TaskRouterEngine

sih_queries = [
    "Find flooded areas under clouds.",
    "Highlight the water body referred to in the query.",
    "What changed between these two dates, and where did the change occur?",
    "Use the optical and SAR images together to identify built-up and water-covered regions.",
    "Has the built-up area increased, decreased, or remained unchanged?",
    "Describe the land-cover and major objects visible in this image."
]

def run_demo():
    router = TaskRouterEngine()
    print("\n" + "=" * 75)
    print("      SATQUERY AI: TASK ROUTER LIVE EVALUATION TEST")
    print("=" * 75 + "\n")

    for i, query in enumerate(sih_queries, 1):
        spec = router.route(query)
        print(f"[{i}] User Query: \"{query}\"")
        print(f"    |-- Task Type:         {spec.task_type.value}")
        print(f"    |-- Modality Required: {spec.modality.value}")
        print(f"    |-- Primary Tool:      {spec.primary_tool.value}")
        print(f"    |-- Sensors:           {[s.value for s in spec.sensors]}")
        print(f"    |-- Bands Required:    {spec.parameters.bands_required}")
        print(f"    |-- Requested Indices: {spec.parameters.indices_requested}")
        print(f"    |-- Evidence Output:   {spec.evidence_requested}")
        print(f"    |-- Confidence:        {int(spec.confidence_threshold * 100)}%")
        print(f"    \\-- Audit Trace:       {spec.audit_summary}")
        print("-" * 75)

    print("\n[SUCCESS] All representative queries successfully parsed and validated against TaskSpec schema!\n")

if __name__ == "__main__":
    run_demo()
