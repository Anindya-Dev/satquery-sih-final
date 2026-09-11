# Task Router & NLP Module (Member 2)

This module handles **SatQuery AI's Natural Language Query Routing & Domain-Specific Adaptation**.

## Responsibilities
- Parse free-form natural language queries from non-expert users into strictly validated JSON schemas.
- Route requests to appropriate downstream models (Single-image VQA/Grounding, Bi-temporal Change Detection, Cross-modal Optical-SAR Fusion).
- Provide fine-tuning scripts (LoRA/QLoRA) for parameter-efficient Small Language Models (SLMs).
- Expose deterministic validation guardrails and fallback execution paths for Member 1 (Data Engine) and Member 3 (GIS Vision Pipeline).
