
# Swarms Models

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)



**Leverage LLM APIs with Unparalleled Speed, Security, and Reliability**


## **Why Swarm Models?**

- **Multi-Provider Support**: Effortlessly integrate APIs from various providers into your projects.

- **Bleeding-Edge Speed**: Experience lightning-fast performance optimized for efficiency.

- **Robust Security**: Built with top-notch security protocols to protect your data and API keys.

- **Ease of Use**: Simple initialization and execution with intuitive `.run(task)` and `__call__` methods.
- **Scalability**: Designed to handle everything from small scripts to large-scale applications.

---

## **Code Example**

```python
from swarm_models import OpenAIChat
import os

# Get the OpenAI API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Create an instance of the OpenAIChat class
model = OpenAIChat(openai_api_key=api_key, model_name="gpt-4o-mini")

# Query the model with a question
out = model(
   "What is the best state to register a business in the US for the least amount of taxes?"
)

# Print the model's response
print(out)
```



## **How It Works**

Swarm Models simplifies the way you interact with different APIs by providing a unified interface for all models.

### **1. Install Swarm Models**

```bash
$ pip3 install swarm-models
```

### **2. Set Your Keys**

```bash
OPENAI_API_KEY="your_openai_api_key"
GROQ_API_KEY="your_groq_api_key"
ANTHROPIC_API_KEY="your_anthropic_api_key"
AZURE_OPENAI_API_KEY="your_azure_openai_api_key"
```

### **3. Initialize a Model**

Import the desired model from the package and initialize it with your API key or necessary configuration.

```python
from swarm_models import YourDesiredModel

model = YourDesiredModel(api_key='your_api_key', *args, **kwargs)
```

### **4. Run Your Task**

Use the `.run(task)` method or simply call the model like `model(task)` with your task.

```python
task = "Define your task here"
result = model.run(task)

# Or equivalently
#result = model(task)
```

### **5. Enjoy the Results**

```python
print(result)
```

---


---


## **Get Started Now**

Ready to streamline your API integrations and boost your application's performance?

1. **Install the Package**

   ```bash
   $ pip install swarm-models
   ```

2. **Explore the Documentation**

   Dive into our comprehensive [Documentation](https://docs.swarms.world) to learn more about the available models and features.

3. **Join the Community**

   Connect with other developers on our [GitHub](https://github.com/swarm-models) and contribute to the project.

[Download Now](https://pypi.org/project/swarm-models/) | [Documentation](https://example.com/docs) | [GitHub](https://github.com/The-Swarm-Corporation/swarm-models)

---

## **Available Models**

| Model Name                | Description                                           |
|---------------------------|-------------------------------------------------------|
| `OpenAIChat`              | Chat model for OpenAI's GPT-3 and GPT-4 APIs.       |
| `Anthropic`               | Model for interacting with Anthropic's APIs.         |
| `AzureOpenAI`             | Azure's implementation of OpenAI's models.           |
| `Dalle3`                  | Model for generating images from text prompts.       |
| `NvidiaLlama31B`         | Llama model for causal language generation.           |
| `Fuyu`                    | Multi-modal model for image and text processing.     |
| `Gemini`                  | Multi-modal model for vision and language tasks.     |
| `Vilt`                    | Vision-and-Language Transformer for question answering.|
| `TogetherLLM`             | Model for collaborative language tasks.               |
| `FireWorksAI`             | Model for generating creative content.                |
| `ReplicateChat`           | Chat model for replicating conversations.             |
| `HuggingfaceLLM`          | Interface for Hugging Face models.                    |
| `CogVLMMultiModal`        | Multi-modal model for vision and language tasks.     |
| `LayoutLMDocumentQA`      | Model for document question answering.                |
| `GPT4VisionAPI`           | Model for analyzing images with GPT-4 capabilities.  |
| `LlamaForCausalLM`        | Causal language model from the Llama family.         |




## **Frequently Asked Questions**

**Q:** *Which providers are supported?*

**A:** Swarm Models supports a wide range of API providers. Check out the [documentation](https://docs.swarms.world) for a full list.

**Q:** *How do I secure my API keys?*

**A:** We recommend using environment variables or a secure key management system. Swarm Models ensures your keys are handled securely within the package.

---

## **Contact Us**

Join our [Discord](https://discord.gg/agora-999382051935506503) to stay updated and get support.
