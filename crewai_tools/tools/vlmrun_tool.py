import requests
from typing import Type
from pydantic.v1 import BaseModel
from crewai_tools.tools.base_tool import BaseTool

class VLMRunInputSchema(BaseModel):
    """Input schema for VLMRunTool."""
    prompt: str = "The prompt for image or document generation."
    generation_type: str = "The type of generation: 'image' or 'document'."

class VLMRunTool(BaseTool):
    name: str = "VLM Run Tool"
    description: str = (
        "This tool uses VLM Run's APIs to generate images or documents based on prompts."
    )
    args_schema: Type[BaseModel] = VLMRunInputSchema

    def _run(self, **kwargs) -> str:
        prompt = kwargs.get("prompt")
        generation_type = kwargs.get("generation_type")

        if not prompt or not generation_type:
            return "Both prompt and generation_type are required."

        if generation_type.lower() not in ["image", "document"]:
            return "Invalid generation_type. Must be 'image' or 'document'."

        api_url = (
            "https://api.vlm.run/v1/image/generate"
            if generation_type.lower() == "image"
            else "https://api.vlm.run/v1/document/generate"
        )

        headers = {
            "Content-Type": "application/json",
            # Add your API key here if required
            # "Authorization": "Bearer YOUR_API_KEY"
        }

        payload = {
            "prompt": prompt,
            # Add any additional parameters required by the API
        }

        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

            # Process the result based on the generation type
            if generation_type.lower() == "image":
                # Assuming the API returns an image URL
                return f"Image generated successfully. URL: {result.get('image_url', 'N/A')}"
            else:
                # Assuming the API returns document content
                return f"Document generated successfully. Content: {result.get('content', 'N/A')}"

        except requests.RequestException as e:
            return f"Error occurred while calling VLM Run API: {str(e)}"
