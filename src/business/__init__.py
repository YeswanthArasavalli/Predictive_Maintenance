"""Phase 2.6 business logic.

Decision- and cost-sensitive validation layer. This package contains only
*deterministic analysis* over already-produced validation predictions — it
trains no model, imports no neural/DL framework, and never reads the official
FD004 test partition. See `src.business.decision`.
"""
