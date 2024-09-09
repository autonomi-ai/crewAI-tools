import pytest
from unittest.mock import patch, MagicMock
from crewai_tools.tools.vlmrun_tool import VLMRunTool
from requests.exceptions import RequestException

@pytest.fixture
def vlm_run_tool():
    return VLMRunTool()

@pytest.mark.parametrize("generation_type, expected_url, expected_result", [
    ("image", "https://api.vlm.run/v1/image/generate", "Image generated successfully"),
    ("document", "https://api.vlm.run/v1/document/generate", "Document generated successfully"),
])
@patch('crewai_tools.tools.vlmrun_tool.requests.post')
def test_generation(mock_post, vlm_run_tool, generation_type, expected_url, expected_result):
    # Mock successful API response
    mock_response = MagicMock()
    mock_response.json.return_value = (
        {"image_url": "https://example.com/image.jpg"} if generation_type == "image"
        else {"content": "Generated document content"}
    )
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    result = vlm_run_tool._run(prompt=f"Test {generation_type} prompt", generation_type=generation_type)

    assert expected_result in result
    assert mock_post.call_args[0][0] == expected_url
    if generation_type == "image":
        assert "https://example.com/image.jpg" in result
    else:
        assert "Generated document content" in result

def test_invalid_generation_type(vlm_run_tool):
    result = vlm_run_tool._run(prompt="Test prompt", generation_type="invalid")
    assert result == "Invalid generation_type. Must be 'image' or 'document'."

def test_missing_parameters(vlm_run_tool):
    result = vlm_run_tool._run(prompt="Test prompt")
    assert result == "Both prompt and generation_type are required."

@patch('crewai_tools.tools.vlmrun_tool.requests.post')
def test_api_error(mock_post, vlm_run_tool):
    # Mock API error
    mock_post.side_effect = RequestException("API Error")

    result = vlm_run_tool._run(prompt="Test prompt", generation_type="image")

    assert "Error occurred while calling VLM Run API" in result
    assert "API Error" in result
