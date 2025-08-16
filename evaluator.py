
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
    GEval
)
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
import asyncio
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from database.models import EvaluationMetrics, SessionLocal
from config.settings import Config

class ChatbotEvaluator:
    def __init__(self):
        self.model = Config.EVALUATION_MODEL

        # Initialize metrics
        self.answer_relevancy = AnswerRelevancyMetric(
            threshold=0.5,
            model=self.model,
            include_reason=True
        )

        self.faithfulness = FaithfulnessMetric(
            threshold=0.7,
            model=self.model,
            include_reason=True
        )

        self.context_precision = ContextualPrecisionMetric(
            threshold=0.5,
            model=self.model,
            include_reason=True
        )

        self.context_recall = ContextualRecallMetric(
            threshold=0.5,
            model=self.model,
            include_reason=True
        )

        self.context_relevancy = ContextualRelevancyMetric(
            threshold=0.5,
            model=self.model,
            include_reason=True
        )

        self.hallucination = HallucinationMetric(
            threshold=0.5,
            model=self.model,
            include_reason=True
        )

        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def calculate_bleu_score(self, reference, candidate):
        """Calculate BLEU score"""
        try:
            # Tokenize the sentences
            reference_tokens = reference.lower().split()
            candidate_tokens = candidate.lower().split()

            # Calculate BLEU score
            bleu_score = sentence_bleu([reference_tokens], candidate_tokens)
            return bleu_score
        except Exception as e:
            print(f"Error calculating BLEU score: {e}")
            return 0.0

    def calculate_rouge_score(self, reference, candidate):
        """Calculate ROUGE score"""
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            # Return ROUGE-L F1 score as representative
            return scores['rougeL'].fmeasure
        except Exception as e:
            print(f"Error calculating ROUGE score: {e}")
            return 0.0

    async def evaluate_response(self, query, response, retrieved_contexts, expected_output=None, transaction_id=None):
        """Evaluate a chatbot response using multiple metrics"""

        # Create test case
        test_case = LLMTestCase(
            input=query,
            actual_output=response,
            expected_output=expected_output or response,  # Use response as expected if not provided
            retrieval_context=[doc.get('content', '') for doc in retrieved_contexts]
        )

        # Initialize results
        results = {
            'answer_relevancy': 0.0,
            'faithfulness': 0.0,
            'context_precision': 0.0,
            'context_recall': 0.0,
            'context_relevancy': 0.0,
            'hallucination_score': 0.0,
            'bleu_score': 0.0,
            'rouge_score': 0.0
        }

        try:
            # Evaluate answer relevancy
            self.answer_relevancy.measure(test_case)
            results['answer_relevancy'] = self.answer_relevancy.score
        except Exception as e:
            print(f"Error evaluating answer relevancy: {e}")

        try:
            # Evaluate faithfulness (only if retrieval context exists)
            if retrieved_contexts:
                self.faithfulness.measure(test_case)
                results['faithfulness'] = self.faithfulness.score
        except Exception as e:
            print(f"Error evaluating faithfulness: {e}")

        try:
            # Evaluate context precision (only if retrieval context exists)
            if retrieved_contexts:
                self.context_precision.measure(test_case)
                results['context_precision'] = self.context_precision.score
        except Exception as e:
            print(f"Error evaluating context precision: {e}")

        try:
            # Evaluate context recall (only if retrieval context exists)
            if retrieved_contexts:
                self.context_recall.measure(test_case)
                results['context_recall'] = self.context_recall.score
        except Exception as e:
            print(f"Error evaluating context recall: {e}")

        try:
            # Evaluate context relevancy (only if retrieval context exists)
            if retrieved_contexts:
                self.context_relevancy.measure(test_case)
                results['context_relevancy'] = self.context_relevancy.score
        except Exception as e:
            print(f"Error evaluating context relevancy: {e}")

        try:
            # Evaluate hallucination
            self.hallucination.measure(test_case)
            results['hallucination_score'] = self.hallucination.score
        except Exception as e:
            print(f"Error evaluating hallucination: {e}")

        # Calculate BLEU score (if expected output is provided)
        if expected_output:
            results['bleu_score'] = self.calculate_bleu_score(expected_output, response)

        # Calculate ROUGE score (if expected output is provided)
        if expected_output:
            results['rouge_score'] = self.calculate_rouge_score(expected_output, response)

        # Store results in database
        if transaction_id:
            self._store_evaluation_results(transaction_id, results)

        return results

    def _store_evaluation_results(self, transaction_id, results):
        """Store evaluation results in database"""
        try:
            db = SessionLocal()

            evaluation_metrics = EvaluationMetrics(
                transaction_id=transaction_id,
                answer_relevancy=results['answer_relevancy'],
                faithfulness=results['faithfulness'],
                context_precision=results['context_precision'],
                context_recall=results['context_recall'],
                context_relevancy=results['context_relevancy'],
                hallucination_score=results['hallucination_score'],
                bleu_score=results['bleu_score'],
                rouge_score=results['rouge_score']
            )

            db.add(evaluation_metrics)
            db.commit()
            db.close()
        except Exception as e:
            print(f"Error storing evaluation results: {e}")

    def evaluate_batch(self, test_cases):
        """Evaluate multiple test cases"""
        metrics = [
            self.answer_relevancy,
            self.faithfulness,
            self.context_precision,
            self.context_recall,
            self.context_relevancy,
            self.hallucination
        ]

        # Run evaluation
        results = evaluate(test_cases, metrics)
        return results

# Utility function to run async evaluation
def evaluate_response_sync(query, response, retrieved_contexts, expected_output=None, transaction_id=None):
    """Synchronous wrapper for async evaluation"""
    evaluator = ChatbotEvaluator()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            evaluator.evaluate_response(query, response, retrieved_contexts, expected_output, transaction_id)
        )
    finally:
        loop.close()
