import os
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from swarms import Agent
from swarm_models import OpenAIChat
import json
import asyncio
from enum import Enum
from swarms.utils.formatter import formatter
from loguru import logger


class AgentRole(Enum):
    SUPERVISOR = "supervisor"
    STATE_DESIGNER = "state_designer"
    LOGIC_HANDLER = "logic_handler"
    VALIDATOR = "validator"


class AgentRequest(BaseModel):
    task: str
    context: Dict[str, Any] = {}


class AppConfig(BaseModel):
    app_description: str
    temperature: float = 0.1
    max_tokens: int = 1000


class StateManager:
    def __init__(self, state_file: str = "app_state.json"):
        self.state_file = state_file
        self._state: Dict[str, Any] = {}
        self._load_state()

    def _load_state(self):
        try:
            with open(self.state_file, "r") as f:
                self._state = json.load(f)
        except FileNotFoundError:
            self._state = {}

    def get_state(self) -> Dict[str, Any]:
        return self._state

    def update_state(self, new_state: Dict[str, Any]):
        self._state = new_state
        self._save_state()

    def _save_state(self):
        with open(self.state_file, "w") as f:
            json.dump(self._state, f, indent=2)


class SpecializedAgent:
    def __init__(self, role: AgentRole, config: AppConfig):
        self.role = role
        self.config = config
        self.agent = self._initialize_agent()

    def _initialize_agent(self) -> Agent:
        system_prompts = {
            AgentRole.SUPERVISOR: """
            You are the supervisor agent responsible for coordinating the backend system. You must ALWAYS return responses in the following JSON format:
            {
                "analysis": {
                    "requirements": [list of identified requirements],
                    "subtasks": [list of subtasks],
                    "priority": string indicating priority level
                },
                "delegation": {
                    "state_designer": {
                        "task": string describing task,
                        "requirements": [specific requirements]
                    },
                    "logic_handler": {
                        "task": string describing task,
                        "requirements": [specific requirements]
                    },
                    "validator": {
                        "task": string describing task,
                        "requirements": [specific requirements]
                    }
                },
                "coordination_plan": {
                    "sequence": [ordered list of steps],
                    "dependencies": [list of dependencies]
                }
            }
            """,
            AgentRole.STATE_DESIGNER: """
            You are the state designer agent responsible for creating and updating application state schemas. You must ALWAYS return responses in the following JSON format:
            {
                "initial_state": {
                    // Dynamic schema based on application type
                    // Example for e-commerce:
                    "products": [
                        {
                            "id": string,
                            "name": string,
                            "price": number,
                            "category": string,
                            // other product fields
                        }
                    ],
                    "categories": [list of category objects],
                    "cart": {
                        "items": [],
                        "total": number
                    }
                },
                "schema_validation": {
                    "is_valid": boolean,
                    "errors": [list of errors if any]
                },
                "sample_data": {
                    // Sample entries matching the schema
                }
            }
            """,
            AgentRole.LOGIC_HANDLER: """
            You are the logic handler agent responsible for implementing business logic. You must ALWAYS return responses in the following JSON format:
            {
                "operation": {
                    "type": string (e.g., "create", "read", "update", "delete"),
                    "status": string (e.g., "success", "failure"),
                    "target": string (entity being operated on)
                },
                "new_state": {
                    // Updated application state
                },
                "response": {
                    "status_code": number,
                    "data": object (operation result),
                    "message": string
                },
                "metadata": {
                    "timestamp": string (ISO format),
                    "operation_id": string
                }
            }
            """,
            AgentRole.VALIDATOR: """
            You are the validator agent responsible for ensuring system correctness. You must ALWAYS return responses in the following JSON format:
            {
                "is_valid": boolean,
                "validation_details": {
                    "schema_validation": {
                        "passed": boolean,
                        "errors": [list of schema errors]
                    },
                    "business_rules": {
                        "passed": boolean,
                        "violations": [list of rule violations]
                    },
                    "data_integrity": {
                        "passed": boolean,
                        "issues": [list of integrity issues]
                    }
                },
                "recommendations": [
                    {
                        "type": string,
                        "description": string,
                        "priority": string
                    }
                ]
            }
            """,
        }

        model = OpenAIChat(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_name=self.config.model_name,
            temperature=self.config.temperature,
        )

        return Agent(
            agent_name=f"{self.role.value}-agent",
            system_prompt=system_prompts[self.role],
            llm=model,
            max_loops=1,
            verbose=True,
            context_length=200000,
            output_type="json",
            formatting_rules={
                "response_format": "json",
                "ensure_json": True,
            },
        )

    async def process(self, request: AgentRequest) -> Dict[str, Any]:
        try:
            result = await asyncio.to_thread(
                self.agent.run,
                json.dumps(
                    {"task": request.task, "context": request.context}
                ),
            )
            return (
                json.loads(result)
                if isinstance(result, str)
                else result
            )
        except Exception as e:
            logger.error(
                f"Error in {self.role.value} agent: {str(e)}"
            )
            raise


class HierarchicalBackend:
    def __init__(self, config: AppConfig):
        self.config = config
        self.state_manager = StateManager()
        self.agents = self._initialize_agents()

    def _initialize_agents(self) -> Dict[AgentRole, SpecializedAgent]:
        return {
            role: SpecializedAgent(role, self.config)
            for role in AgentRole
        }

    async def initialize_application(
        self, app_description: str
    ) -> Dict[str, Any]:
        """Initialize the application state based on description"""
        supervisor = self.agents[AgentRole.SUPERVISOR]
        state_designer = self.agents[AgentRole.STATE_DESIGNER]
        validator = self.agents[AgentRole.VALIDATOR]

        # Supervisor analyzes requirements
        supervisor_result = await supervisor.process(
            AgentRequest(
                task="Analyze application requirements and create task breakdown",
                context={"app_description": app_description},
            )
        )

        # State designer creates initial state
        state_design = await state_designer.process(
            AgentRequest(
                task="Create initial state schema and sample data",
                context={
                    "app_description": app_description,
                    "requirements": supervisor_result,
                },
            )
        )

        # Validator checks the state design
        validation_result = await validator.process(
            AgentRequest(
                task="Validate initial state design",
                context={"state_design": state_design},
            )
        )

        if validation_result.get("is_valid", False):
            self.state_manager.update_state(
                state_design["initial_state"]
            )
            return state_design
        else:
            raise ValueError(
                f"State validation failed: {validation_result.get('errors')}"
            )

    async def process_request(
        self,
        endpoint: str,
        method: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process API requests using the agent hierarchy"""
        current_state = self.state_manager.get_state()

        # Supervisor coordinates request handling
        supervisor_result = await self.agents[
            AgentRole.SUPERVISOR
        ].process(
            AgentRequest(
                task="Analyze API request and coordinate processing",
                context={
                    "endpoint": endpoint,
                    "method": method,
                    "params": params,
                    "current_state": current_state,
                },
            )
        )

        # Logic handler processes the request
        logic_result = await self.agents[
            AgentRole.LOGIC_HANDLER
        ].process(
            AgentRequest(
                task="Process API request and update state",
                context={
                    "request_analysis": supervisor_result,
                    "current_state": current_state,
                },
            )
        )

        # Validator verifies the response
        validation_result = await self.agents[
            AgentRole.VALIDATOR
        ].process(
            AgentRequest(
                task="Validate response and state changes",
                context={
                    "original_state": current_state,
                    "new_state": logic_result.get("new_state"),
                    "response": logic_result.get("response"),
                },
            )
        )

        if validation_result.get("is_valid", False):
            if "new_state" in logic_result:
                self.state_manager.update_state(
                    logic_result["new_state"]
                )
            return logic_result
        else:
            raise HTTPException(
                status_code=400,
                detail=validation_result.get("errors"),
            )


def create_app(config: AppConfig) -> FastAPI:
    formatter.print_panel(
        "Creating FastAPI app with title: Hierarchical LLM-Powered Backend"
    )
    app = FastAPI(
        title="Hierarchical LLM-Powered Backend", debug=True
    )
    formatter.print_panel(
        f"FastAPI app created with debug mode: {app.debug}"
    )
    backend = HierarchicalBackend(config)
    formatter.print_panel(
        f"HierarchicalBackend instance created with config: {config}"
    )

    @app.post("/initialize")
    async def initialize_app(app_description: str):
        """Initialize the application with a description"""
        formatter.print_panel(
            f"Initializing application with description: {app_description}"
        )
        result = await backend.initialize_application(app_description)
        formatter.print_panel(
            f"Application initialized with result: {result}"
        )
        return result

    @app.post("/api/{path:path}")
    async def dynamic_endpoint(path: str, request: Dict[str, Any]):
        formatter.print_panel(
            f"Processing POST request for path: {path} with request: {request}"
        )
        result = await backend.process_request(path, "POST", request)
        formatter.print_panel(
            f"POST request processed with result: {result}"
        )
        return result

    @app.get("/api/{path:path}")
    async def dynamic_get_endpoint(path: str):
        formatter.print_panel(
            f"Processing GET request for path: {path}"
        )
        result = await backend.process_request(path, "GET")
        formatter.print_panel(
            f"GET request processed with result: {result}"
        )
        return result

    formatter.print_panel("FastAPI app routes defined")
    return app


# Example usage
if __name__ == "__main__":
    import uvicorn

    config = AppConfig(
        app_description="Initialize with API call",
    )

    app = create_app(config)

    logger.add("app.log", rotation="10 MB")

    uvicorn.run(app, host="0.0.0.0", port=8000)
    formatter.print_panel(
        "Uvicorn server started on host: 0.0.0.0 and port: 8000"
    )
