"""
A-MEM Evaluator Module

This module implements the Evaluator agent from the harness architecture,
providing quality assessment for memory notes before they enter the memory network.
"""

from typing import Dict, List, Optional, Any, Tuple
import json
import logging

logger = logging.getLogger(__name__)


class NoteEvaluator:
    """
    Evaluator for assessing memory note quality before storage.

    Implements 6 evaluation dimensions:
    1. Completeness - Does the note capture all key information?
    2. Credibility - Is the information trustworthy?
    3. Relevance - How relevant is this to current context?
    4. Uniqueness - Is this distinct from existing memories?
    5. Actionability - Can this memory be effectively retrieved later?
    6. Consistency - Is this consistent with existing memory network?
    """

    # Evaluation thresholds for each dimension
    EVALUATION_THRESHOLDS = {
        "completeness": 6,
        "credibility": 7,
        "relevance": 5,
        "uniqueness": 5,
        "actionability": 6,
        "consistency": 6,
    }

    # Dimension weights for overall score calculation
    EVALUATION_WEIGHTS = {
        "completeness": 0.15,
        "credibility": 0.20,
        "relevance": 0.15,
        "uniqueness": 0.10,
        "actionability": 0.20,
        "consistency": 0.20,
    }

    # System prompt for the evaluator
    SYSTEM_PROMPT = """You are an AI memory quality evaluator for A-MEM, an agentic memory system.
Your role is to critically assess each memory note before it enters the memory network.

EVALUATION DIMENSIONS:
1. Completeness (0-10): Does the note capture all key information?
   - Score 10: Contains full context, specific details, and actionable information
   - Score 5: Partial information, missing some key details
   - Score 0: Vague, lacks substance

2. Credibility (0-10): Is the information trustworthy?
   - Score 10: Factual, verifiable information from trusted source
   - Score 5: Likely true but cannot verify
   - Score 0: Contradicts known facts or seems unreliable

3. Relevance (0-10): How relevant is this to the current context/task?
   - Score 10: Highly relevant, directly applicable to current situation
   - Score 5: Somewhat relevant, tangential
   - Score 0: Irrelevant to current situation

4. Uniqueness (0-10): Is this information distinct from existing memories?
   - Score 10: Adds new information not in memory network
   - Score 5: Partially overlaps with existing memories
   - Score 0: Exact duplicate of existing memory

5. Actionability (0-10): Can this memory be effectively retrieved and used later?
   - Score 10: Clear, specific, well-structured for retrieval
   - Score 5: Somewhat vague but retrievable with effort
   - Score 0: Too vague or unstructured to retrieve

6. Consistency (0-10): Is this consistent with the existing memory network?
   - Score 10: Perfectly aligns with related memories
   - Score 5: Minor conflicts that can be reconciled
   - Score 0: Directly contradicts established memories

IMPORTANT: You must be critical in your evaluation. Do NOT give inflated scores.
LLMs tend to be overly positive when evaluating their own outputs. Be strict and honest.

OUTPUT FORMAT:
You must respond with a valid JSON object containing:
{
    "scores": {
        "completeness": {"score": int, "reasoning": "brief explanation"},
        "credibility": {"score": int, "reasoning": "brief explanation"},
        "relevance": {"score": int, "reasoning": "brief explanation"},
        "uniqueness": {"score": int, "reasoning": "brief explanation"},
        "actionability": {"score": int, "reasoning": "brief explanation"},
        "consistency": {"score": int, "reasoning": "brief explanation"}
    },
    "overall_score": float,
    "decision": "ACCEPT" | "REJECT" | "REVISE",
    "revision_suggestions": [
        {
            "dimension": "dimension_name",
            "current_value": "what it currently has",
            "suggestion": "specific improvement suggestion"
        }
    ],
    "confidence": float
}"""

    def __init__(self, llm_controller):
        """
        Initialize the evaluator.

        Args:
            llm_controller: LLM controller for generating evaluations
        """
        self.llm_controller = llm_controller

    def evaluate(
        self,
        note_content: str,
        note_context: str,
        note_keywords: List[str],
        note_tags: List[str],
        related_memories: List[Dict[str, Any]] = None,
        current_task: str = None
    ) -> Dict[str, Any]:
        """
        Evaluate a memory note across all dimensions.

        Args:
            note_content: The main content of the note
            note_context: The context/summary of the note
            note_keywords: Extracted keywords
            note_tags: Assigned tags
            related_memories: List of related existing memories for consistency check
            current_task: Current task context (optional)

        Returns:
            Evaluation result with scores, decision, and suggestions
        """
        # Build the prompt with note details and context
        prompt = self._build_evaluation_prompt(
            note_content, note_context, note_keywords, note_tags,
            related_memories, current_task
        )

        try:
            # Get evaluation from LLM
            response = self.llm_controller.get_completion(
                prompt,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "evaluation",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "scores": {
                                    "type": "object",
                                    "properties": {
                                        "completeness": {
                                            "type": "object",
                                            "properties": {
                                                "score": {"type": "integer"},
                                                "reasoning": {"type": "string"}
                                            }
                                        },
                                        "credibility": {
                                            "type": "object",
                                            "properties": {
                                                "score": {"type": "integer"},
                                                "reasoning": {"type": "string"}
                                            }
                                        },
                                        "relevance": {
                                            "type": "object",
                                            "properties": {
                                                "score": {"type": "integer"},
                                                "reasoning": {"type": "string"}
                                            }
                                        },
                                        "uniqueness": {
                                            "type": "object",
                                            "properties": {
                                                "score": {"type": "integer"},
                                                "reasoning": {"type": "string"}
                                            }
                                        },
                                        "actionability": {
                                            "type": "object",
                                            "properties": {
                                                "score": {"type": "integer"},
                                                "reasoning": {"type": "string"}
                                            }
                                        },
                                        "consistency": {
                                            "type": "object",
                                            "properties": {
                                                "score": {"type": "integer"},
                                                "reasoning": {"type": "string"}
                                            }
                                        }
                                    },
                                    "required": ["completeness", "credibility", "relevance",
                                                "uniqueness", "actionability", "consistency"]
                                },
                                "overall_score": {"type": "number"},
                                "decision": {"type": "string", "enum": ["ACCEPT", "REJECT", "REVISE"]},
                                "revision_suggestions": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "dimension": {"type": "string"},
                                            "current_value": {"type": "string"},
                                            "suggestion": {"type": "string"}
                                        }
                                    }
                                },
                                "confidence": {"type": "number"}
                            },
                            "required": ["scores", "overall_score", "decision",
                                        "revision_suggestions", "confidence"]
                        }
                    }
                }
            )

            evaluation = json.loads(response)
            return self._process_evaluation(evaluation, related_memories)

        except Exception as e:
            logger.error(f"Error in evaluation: {str(e)}")
            # Return a safe fallback that accepts the note
            return self._get_fallback_evaluation()

    def _build_evaluation_prompt(
        self,
        note_content: str,
        note_context: str,
        note_keywords: List[str],
        note_tags: List[str],
        related_memories: List[Dict[str, Any]] = None,
        current_task: str = None
    ) -> str:
        """Build the evaluation prompt with context."""

        prompt_parts = [
            self.SYSTEM_PROMPT,
            "\n\nNOTE TO EVALUATE:",
            f"\nContent: {note_content}",
            f"\nContext: {note_context}",
            f"\nKeywords: {', '.join(note_keywords) if note_keywords else 'None'}",
            f"\nTags: {', '.join(note_tags) if note_tags else 'None'}"
        ]

        if current_task:
            prompt_parts.append(f"\n\nCURRENT TASK CONTEXT: {current_task}")

        if related_memories and len(related_memories) > 0:
            prompt_parts.append("\n\nRELATED EXISTING MEMORIES (for consistency check):")
            for i, mem in enumerate(related_memories[:5], 1):  # Limit to 5 for context
                prompt_parts.append(
                    f"\n{i}. [{mem.get('id', 'unknown')}] "
                    f"Content: {mem.get('content', '')[:200]}... "
                    f"Context: {mem.get('context', 'N/A')} "
                    f"Tags: {', '.join(mem.get('tags', []))}"
                )
        else:
            prompt_parts.append("\n\nNo related existing memories found.")

        prompt_parts.append("\n\nAnalyze the note and provide your critical evaluation:")

        return "".join(prompt_parts)

    def _process_evaluation(
        self,
        evaluation: Dict,
        related_memories: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process and validate the evaluation result."""

        scores = evaluation.get("scores", {})
        score_values = {
            "completeness": scores.get("completeness", {}).get("score", 0),
            "credibility": scores.get("credibility", {}).get("score", 0),
            "relevance": scores.get("relevance", {}).get("score", 0),
            "uniqueness": scores.get("uniqueness", {}).get("score", 0),
            "actionability": scores.get("actionability", {}).get("score", 0),
            "consistency": scores.get("consistency", {}).get("score", 0),
        }

        # Calculate weighted overall score
        overall = sum(
            score_values[k] * self.EVALUATION_WEIGHTS[k]
            for k in score_values
        )

        # Determine decision based on rules
        decision = self._determine_decision(score_values, overall)

        # Build result
        result = {
            "scores": scores,
            "score_values": score_values,
            "overall_score": round(overall, 2),
            "decision": decision,
            "revision_suggestions": evaluation.get("revision_suggestions", []),
            "confidence": evaluation.get("confidence", 0.5)
        }

        return result

    def _determine_decision(self, scores: Dict[str, int], overall: float) -> str:
        """
        Determine the evaluation decision based on scores.

        Rules:
        - ACCEPT: majority of dimensions above threshold AND overall >= 6.0
        - REVISE: 2-3 dimensions below threshold OR overall >= 5.0
        - REJECT: more than 3 dimensions below threshold OR overall < 5.0
        """
        below_threshold = sum(
            1 for dim, score in scores.items()
            if score < self.EVALUATION_THRESHOLDS.get(dim, 5)
        )

        if below_threshold == 0 and overall >= 6.0:
            return "ACCEPT"
        elif below_threshold <= 2 and overall >= 5.5:
            return "REVISE"
        else:
            return "REJECT"

    def _get_fallback_evaluation(self) -> Dict[str, Any]:
        """Return a safe fallback evaluation when LLM fails."""
        return {
            "scores": {
                "completeness": {"score": 5, "reasoning": "Fallback due to evaluation error"},
                "credibility": {"score": 5, "reasoning": "Fallback due to evaluation error"},
                "relevance": {"score": 5, "reasoning": "Fallback due to evaluation error"},
                "uniqueness": {"score": 5, "reasoning": "Fallback due to evaluation error"},
                "actionability": {"score": 5, "reasoning": "Fallback due to evaluation error"},
                "consistency": {"score": 5, "reasoning": "Fallback due to evaluation error"}
            },
            "score_values": {
                "completeness": 5,
                "credibility": 5,
                "relevance": 5,
                "uniqueness": 5,
                "actionability": 5,
                "consistency": 5
            },
            "overall_score": 5.0,
            "decision": "REVISE",
            "revision_suggestions": [
                {
                    "dimension": "all",
                    "current_value": "evaluation failed",
                    "suggestion": "Please review the note content manually"
                }
            ],
            "confidence": 0.0
        }

    def get_evaluation_summary(self, evaluation: Dict) -> str:
        """Generate a human-readable summary of the evaluation."""
        score_values = evaluation.get("score_values", {})

        summary_lines = [
            "=== Note Evaluation Summary ===",
            f"Overall Score: {evaluation.get('overall_score', 0):.1f}/10",
            f"Decision: {evaluation.get('decision', 'UNKNOWN')}",
            f"Confidence: {evaluation.get('confidence', 0)*100:.0f}%",
            "",
            "Dimension Scores:"
        ]

        for dim, score in score_values.items():
            threshold = self.EVALUATION_THRESHOLDS.get(dim, 5)
            status = "✓" if score >= threshold else "✗"
            summary_lines.append(
                f"  {status} {dim.capitalize()}: {score}/10 (threshold: {threshold})"
            )

        suggestions = evaluation.get("revision_suggestions", [])
        if suggestions:
            summary_lines.append("")
            summary_lines.append("Revision Suggestions:")
            for sug in suggestions:
                summary_lines.append(
                    f"  - [{sug.get('dimension', 'unknown')}] {sug.get('suggestion', '')}"
                )

        return "\n".join(summary_lines)


class RevisionAgent:
    """
    Agent responsible for revising notes based on evaluation feedback.
    Implements the "Generator" role from the harness architecture.
    """

    SYSTEM_PROMPT = """You are an AI memory revision agent for A-MEM.
Your role is to improve memory notes based on evaluation feedback.

Given a memory note and revision suggestions, you will:
1. Analyze the current note content
2. Apply the suggested improvements
3. Generate an enhanced version of the note

Return the revised note in JSON format with the improved fields."""

    def __init__(self, llm_controller):
        """Initialize the revision agent."""
        self.llm_controller = llm_controller

    def revise(
        self,
        note_content: str,
        note_context: str,
        note_keywords: List[str],
        note_tags: List[str],
        revision_suggestions: List[Dict]
    ) -> Dict[str, Any]:
        """
        Revise a note based on feedback.

        Args:
            note_content: Original content
            note_context: Original context
            note_keywords: Original keywords
            note_tags: Original tags
            revision_suggestions: List of revision suggestions from evaluator

        Returns:
            Revised note fields
        """
        prompt = self._build_revision_prompt(
            note_content, note_context, note_keywords, note_tags,
            revision_suggestions
        )

        try:
            response = self.llm_controller.get_completion(
                prompt,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "revision",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "context": {"type": "string"},
                                "keywords": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            )

            return json.loads(response)

        except Exception as e:
            logger.error(f"Error in revision: {str(e)}")
            # Return original if revision fails
            return {
                "content": note_content,
                "context": note_context,
                "keywords": note_keywords,
                "tags": note_tags
            }

    def _build_revision_prompt(
        self,
        note_content: str,
        note_context: str,
        note_keywords: List[str],
        note_tags: List[str],
        revision_suggestions: List[Dict]
    ) -> str:
        """Build the revision prompt."""

        suggestions_text = []
        for sug in revision_suggestions:
            suggestions_text.append(
                f"- [{sug.get('dimension', 'unknown')}] "
                f"Current: {sug.get('current_value', 'N/A')}, "
                f"Suggestion: {sug.get('suggestion', '')}"
            )

        prompt = f"""{self.SYSTEM_PROMPT}

ORIGINAL NOTE:
Content: {note_content}
Context: {note_context}
Keywords: {', '.join(note_keywords) if note_keywords else 'None'}
Tags: {', '.join(note_tags) if note_tags else 'None'}

REVISION SUGGESTIONS:
{chr(10).join(suggestions_text)}

Please provide the revised note with improvements applied.
Return as JSON with fields: content, context, keywords, tags"""

        return prompt
