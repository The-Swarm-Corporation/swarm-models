from swarm_models.sam_two import GroundedSAMTwo
from loguru import logger

# Example usage:
ontology = {"shipping container": "container"}
runner = GroundedSAMTwo(ontology)

# Run on a single image
image_path = "path/to/your/image.jpg"
json_output = runner.run(image_path, output_dir="annotated_images")
logger.info("Annotation result: \n{}", json_output)

# Run on a dataset (directory)
image_dir = "path/to/your/dataset"
json_output = runner.run(image_dir)
logger.info("Dataset labeling result: \n{}", json_output)
