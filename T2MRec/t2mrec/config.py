"""Default configuration for T2MRec."""

from dataclasses import dataclass


@dataclass(frozen=True)
class T2MRecDefaults:
    embedding_type: str = "tfidf"
    semantic_weight: float = 0.9
    structural_weight: float = 0.1
    epochs: int = 200
    learning_rate: float = 1e-3
    hidden_dim: int = 512
    output_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.2
    batch_size: int = 256
    temperature: float = 0.07
    anchor_k: int = 5
    expansion_k: int = 10
    final_k: int = 10
    llm_model: str = "gpt-5.2"


DEFAULTS = T2MRecDefaults()
