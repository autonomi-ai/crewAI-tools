import unittest
from unittest.mock import patch, MagicMock
from crewai_tools.tools.vlmrun_tool import VLMRunTool

class TestVLMRunTool(unittest.TestCase):
    def setUp(self):
        self.tool = VLMRunTool()

    @patch('crewai_tools.tools.vlmrun_tool.requests.post')
    def test_image_generation(self, mock_post):
        # Mock successful API response for image generation
        mock_response = MagicMock()
        mock_response.json.return_value = {"image_url": "https://example.com/image.jpg"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.tool._run(prompt="A beautiful sunset", generation_type="image")
        self.assertIn("Image generated successfully", result)
        self.assertIn("https://example.com/image.jpg", result)

    @patch('crewai_tools.tools.vlmrun_tool.requests.post')
    def test_document_generation(self, mock_post):
        # Mock successful API response for document generation
        mock_response = MagicMock()
        mock_response.json.return_value = {"content": "Generated document content"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = self.tool._run(prompt="Write a short story", generation_type="document")
        self.assertIn("Document generated successfully", result)
        self.assertIn("Generated document content", result)

    def test_invalid_generation_type(self):
        result = self.tool._run(prompt="Test prompt", generation_type="invalid")
        self.assertEqual(result, "Invalid generation_type. Must be 'image' or 'document'.")

    def test_missing_parameters(self):
        result = self.tool._run(prompt="Test prompt")
        self.assertEqual(result, "Both prompt and generation_type are required.")

    @patch('crewai_tools.tools.vlmrun_tool.requests.post')
    def test_api_error(self, mock_post):
        # Mock API error
        mock_post.side_effect = Exception("API Error")

        result = self.tool._run(prompt="Test prompt", generation_type="image")
        self.assertIn("Error occurred while calling VLM Run API", result)

if __name__ == '__main__':
    unittest.main()
