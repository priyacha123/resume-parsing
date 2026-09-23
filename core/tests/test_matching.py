from django.test import TestCase
from core.matching import compute_tfidf_score


class TFIDFMatchingTests(TestCase):
    def test_identical_text_scores_high(self):
        text = "Python Django REST API PostgreSQL"
        score = compute_tfidf_score(text, text)
        self.assertGreater(score, 90)

    def test_unrelated_text_scores_low(self):
        resume = "Python Django backend developer PostgreSQL Redis"
        jd = "Head chef needed for fine dining restaurant, pastry experience"
        score = compute_tfidf_score(resume, jd)
        self.assertLess(score, 20)

    def test_empty_text_returns_zero(self):
        score = compute_tfidf_score("", "some job description")
        self.assertEqual(score, 0.0)