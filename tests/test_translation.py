import pytest
from unittest.mock import MagicMock
from app.services.translation import TranslationService
from app.services.llm import LLMService

@pytest.fixture
def mock_llm():
    return MagicMock(spec=LLMService)

@pytest.fixture
def translation_service(mock_llm):
    return TranslationService(llm_service=mock_llm)

def test_detect_language_french(translation_service, mock_llm):
    mock_llm.generate.return_value = "French"
    lang = translation_service.detect_language("Bonjour, comment ça va?")
    assert lang == "French"
    mock_llm.generate.assert_called_once()

def test_detect_language_fallback(translation_service, mock_llm):
    mock_llm.generate.side_effect = Exception("API connection timed out")
    lang = translation_service.detect_language("Bonjour")
    assert lang == "English" # Default fallback

def test_translate_to_english(translation_service, mock_llm):
    mock_llm.generate.return_value = "What is the policy on maternity leave?"
    translation = translation_service.translate_to_english("Quelle est la politique sur le congé maternité?", "French")
    assert translation == "What is the policy on maternity leave?"

def test_translate_from_english(translation_service, mock_llm):
    mock_llm.generate.return_value = "L'employé a droit à 20 jours de congé [1]."
    translation = translation_service.translate_from_english("The employee is entitled to 20 days of leave [1].", "French")
    assert "jours de congé" in translation
    assert "[1]" in translation
