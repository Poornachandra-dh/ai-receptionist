import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import os
import shutil
import unittest
from rag.ingest import DocumentIngester
from rag.retriever import RAGRetriever

class TestRAGPipeline(unittest.TestCase):
    test_dir = "./data/test_rag"
    sample_file = "./data/sample_clinic_faq.md"

    @classmethod
    def setUpClass(cls):
        Path(cls.test_dir).mkdir(parents=True, exist_ok=True)
        cls.ingester = DocumentIngester(data_dir=cls.test_dir)
        cls.ingest_result = cls.ingester.ingest_file(cls.sample_file)
        cls.retriever = RAGRetriever(data_dir=cls.test_dir, similarity_threshold=0.48)

    @classmethod
    def tearDownClass(cls):
        if Path(cls.test_dir).exists():
            shutil.rmtree(cls.test_dir)

    def test_ingestion_success(self):
        self.assertGreater(self.ingest_result["total_vectors"], 0)
        self.assertGreater(self.ingest_result["new_chunks_added"], 0)

    def test_in_scope_retrieval(self):
        result = self.retriever.retrieve("What are your hours on Saturday?")
        self.assertTrue(result["has_context"], "Expected in-scope query to find context")
        self.assertGreater(result["max_score"], 0.50)
        self.assertIn("Saturday", result["formatted_context"])

    def test_out_of_scope_hallucination_prevention(self):
        result = self.retriever.retrieve("What is the current stock price of Tesla or Bitcoin?")
        self.assertFalse(result["has_context"], "Out-of-scope query should not match clinic context")
        self.assertEqual(result["reason"], "LOW_CONFIDENCE")

    def test_empty_query(self):
        result = self.retriever.retrieve("   ")
        self.assertFalse(result["has_context"])
        self.assertEqual(result["reason"], "EMPTY_QUERY")

if __name__ == "__main__":
    unittest.main()
