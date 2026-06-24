"""
Agentic research system.

Given a high-level goal, the ResearchAgent:
  1. Decomposes the goal into 3-5 specific sub-questions (Gemini)
  2. Answers each sub-question against the RAG pipeline
  3. Synthesizes all findings into a cohesive report (Gemini)
"""
import json
import os
import re
import sys
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "PDF_summarizer"))

from google import genai
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

from rag_gemini import GeminiRAGPipeline, RetrievalFilters

DECOMPOSE_MODEL = "models/gemini-3.5-flash"
SYNTHESIS_MODEL = "models/gemini-3.5-flash"


class ResearchAgent:
    def __init__(self, rag: GeminiRAGPipeline):
        self.rag = rag
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        self.client = genai.Client(api_key=api_key)

    def run(
        self,
        goal: str,
        top_k: int = 3,
        filters: Optional[RetrievalFilters] = None,
    ) -> dict:
        """
        Run the full agentic pipeline.

        Returns:
            {
                "goal": str,
                "sub_queries": [{"question": str, "answer": str, "chunks_used": [...]}],
                "synthesis": str,
            }
        """
        sub_questions = self._decompose(goal)
        sub_results = []
        for question in sub_questions:
            result = self.rag.answer_question(question, top_k=top_k, filters=filters)
            sub_results.append({
                "question": question,
                "answer": result["answer"],
                "chunks_used": result["chunks_used"],
            })

        synthesis = self._synthesize(goal, sub_results)

        return {
            "goal": goal,
            "sub_queries": sub_results,
            "synthesis": synthesis,
        }

    def _decompose(self, goal: str) -> List[str]:
        """Ask Gemini to break the goal into 3-5 specific sub-questions."""
        prompt = (
            "You are a senior financial research analyst.\n"
            "Given the research goal below, generate 3 to 5 specific, self-contained questions "
            "that together would fully address the goal. Each question should be answerable from "
            "financial documents (reports, filings, earnings calls).\n\n"
            f"Research goal: {goal}\n\n"
            "Return ONLY a JSON array of question strings, with no additional text or markdown. "
            "Example: [\"What was revenue in Q3?\", \"What are the main risk factors?\"]"
        )

        response = self.client.models.generate_content(model=DECOMPOSE_MODEL, contents=prompt, config={"temperature": 0})
        raw = response.text.strip() if hasattr(response, "text") else str(response).strip()

        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            questions = json.loads(raw)
            if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
                return questions[:5]
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: extract quoted strings
        questions = re.findall(r'"([^"]{10,})"', raw)
        if questions:
            return questions[:5]

        # Last resort: treat entire goal as single question
        return [goal]

    def _synthesize(self, goal: str, sub_results: List[dict]) -> str:
        """Ask Gemini to synthesize all sub-answers into a final report."""
        findings = "\n\n".join(
            f"Sub-question {i+1}: {r['question']}\nAnswer: {r['answer']}"
            for i, r in enumerate(sub_results)
        )

        prompt = (
            "You are a senior equity research analyst writing a client report.\n"
            "Using ONLY the research findings below, write a cohesive analytical report "
            f"addressing this goal: {goal}\n\n"
            "Your report should:\n"
            "- Synthesize the findings into a coherent narrative\n"
            "- Highlight the most important insights\n"
            "- Note any gaps or areas of uncertainty\n"
            "- Be concise (3-5 paragraphs)\n\n"
            f"Research findings:\n{findings}\n\n"
            "Write the synthesis report:"
        )

        response = self.client.models.generate_content(model=SYNTHESIS_MODEL, contents=prompt, config={"temperature": 0})
        return response.text.strip() if hasattr(response, "text") else str(response).strip()
