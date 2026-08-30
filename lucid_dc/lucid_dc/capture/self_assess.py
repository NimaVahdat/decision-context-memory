"""Self-assessment: removed.

All trace analysis is now done by the LLM analyzer in
consolidate.py. The heuristic keyword matching that was here
created false positives (e.g., "error" in fixture notes
triggered hallucination_risk) and polluted the correction
chain with synthetic corrections that drowned real human
feedback.
"""
