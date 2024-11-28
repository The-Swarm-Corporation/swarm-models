from swarm_models.lite_llm_model import LiteLLMModel

model = LiteLLMModel()
output = model.run("hey")
print(output)