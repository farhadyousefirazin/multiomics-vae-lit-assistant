import boto3


class BedrockLLMClient:
    def __init__(
        self,
        model_id: str = "eu.amazon.nova-pro-v1:0",
        region_name: str = "eu-north-1",
    ):
        self.model_id = model_id
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region_name,
        )

    def generate_answer(
        self,
        prompt: str,
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> str:
        response = self.client.converse(
            modelId=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ],
            inferenceConfig={
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        )

        return response["output"]["message"]["content"][0]["text"]