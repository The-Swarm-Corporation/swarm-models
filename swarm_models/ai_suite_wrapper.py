import aisuite as ai


class AISuiteWrapper:
    """
    A wrapper class for the AISuite client.
    """

    def __init__(
        self,
        model_name: str = "anthropic:claude-3-5-sonnet-20240620",
        system_prompt: str = None,
        temperature: float = None,
        user_name: str = None,
    ):
        """
        Initialize the AISuiteWrapper with the model name, system prompt, and user prompt.
        """
        self.client = ai.Client()
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.user_name = user_name

    def run(self, task: str, *args, **kwargs):
        """
        Run the AISuite model with the given task.
        """
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            temperature=self.temperature,
            *args,
            **kwargs,
        )

        return response.choices[0].message.content

    def __call__(self, task: str = None, *args, **kwargs):
        return self.run(task, *args, **kwargs)


model = AISuiteWrapper()
print(model.run("what is quantum field theory about"))
