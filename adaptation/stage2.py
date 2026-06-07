"""
Backward-compatible stage2 interface that re-exports the modular pipelines.
"""

from .base_trainer import BaseAdaptationTrainer
from .node_level import (
    NodeLevelAdaptationTrainer,
    evaluate_node_level_stage,
    print_node_level_results_summary,
)
from .one_shot import evaluate_node_level_one_shot_stage
from .tabpfn_eval import evaluate_node_level_tabpfn_stage

# Backward-compatible aliases for legacy imports
NodeAdaptationTrainer = NodeLevelAdaptationTrainer
evaluate_node_classification_stage = evaluate_node_level_stage
print_results_summary = print_node_level_results_summary

# Preserve legacy name for downstream imports.
AdpationTrainer = NodeAdaptationTrainer

__all__ = [
    "BaseAdaptationTrainer",
    "NodeLevelAdaptationTrainer",
    "NodeAdaptationTrainer",
    "AdpationTrainer",
    "evaluate_node_level_stage",
    "evaluate_node_level_one_shot_stage",
    "evaluate_node_level_tabpfn_stage",
    "evaluate_node_classification_stage",
    "print_node_level_results_summary",
    "print_results_summary",
]
