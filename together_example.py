import os
from swarm_models import TogetherLLM


# Example usage
if __name__ == "__main__":
    model_runner = TogetherLLM(
        api_key=os.environ.get("TOGETHER_API_KEY"),
        model_name="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        system_prompt="You're Larry fink",
    )
    tasks = [
        "What are the top-performing mutual funds in the last quarter?",
        "How do I evaluate the risk of a mutual fund?",
        "What are the fees associated with investing in a mutual fund?",
        "Can you recommend a mutual fund for a beginner investor?",
        "How do I diversify my portfolio with mutual funds?",
    ]
    # response_contents = model_runner.run_concurrently(tasks)
    # for response_content in response_contents:
    #     print(response_content)
    print(
        model_runner.run(
            "How do we allocate capital efficiently in your opion Larry?"
        )
    )
