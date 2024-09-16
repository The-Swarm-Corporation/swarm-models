
# Swarms Models

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)



**Leverage APIs with Unparalleled Speed and Security**


## **Why Swarm Models?**

- **Multi-Provider Support**: Effortlessly integrate APIs from various providers into your projects.

- **Bleeding-Edge Speed**: Experience lightning-fast performance optimized for efficiency.

- **Robust Security**: Built with top-notch security protocols to protect your data and API keys.

- **Ease of Use**: Simple initialization and execution with intuitive `.run(task)` and `__call__` methods.
- **Scalability**: Designed to handle everything from small scripts to large-scale applications.

---

## **How It Works**

Swarm Models simplifies the way you interact with different APIs by providing a unified interface for all models.

### **1. Install Swarm Models**

```bash
$ pip3 install swarm-models
```

### **2. Initialize a Model**

Import the desired model from the package and initialize it with your API key or necessary configuration.

```python
from swarm_models import YourDesiredModel

model = YourDesiredModel(api_key='your_api_key')
```

### **3. Run Your Task**

Use the `.run(task)` method or simply call the model with your task.

```python
task = "Define your task here"
result = model.run(task)
# Or equivalently
result = model(task)
```

### **4. Enjoy the Results**

```python
print(result)
```

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

---


## **Get Started Now**

Ready to streamline your API integrations and boost your application's performance?

1. **Install the Package**

   ```bash
   $ pip install swarm-models
   ```

2. **Explore the Documentation**

   Dive into our comprehensive [Documentation](https://example.com/docs) to learn more about the available models and features.

3. **Join the Community**

   Connect with other developers on our [GitHub](https://github.com/swarm-models) and contribute to the project.

[Download Now](https://pypi.org/project/swarm-models/) | [Documentation](https://example.com/docs) | [GitHub](https://github.com/swarm-models)

---

## **Frequently Asked Questions**

**Q:** *Which providers are supported?*

**A:** Swarm Models supports a wide range of API providers. Check out the [documentation](https://example.com/docs/providers) for a full list.

**Q:** *How do I secure my API keys?*

**A:** We recommend using environment variables or a secure key management system. Swarm Models ensures your keys are handled securely within the package.

---

## **Contact Us**

Join our [Discord](https://discord.gg/agora-999382051935506503) to stay updated and get support.
