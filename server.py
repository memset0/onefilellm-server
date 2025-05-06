import dotenv

import os
import asyncio
from fastapi import FastAPI, Response
from agent import Agent, AgentTask
import uvicorn

app = FastAPI()
agent = Agent()


def get_config(key, default_value=None):
    if not key.startswith('OFL_'):
        key = 'OFL_' + key
    return os.getenv(key.upper(), default_value)


@app.get("/{task_description:path}")
@app.post("/{task_description:path}")
async def add_task_route(task_description: str):
    if task_description == "favicon.ico":
        return Response(content="Not found", status_code=404)

    input_path = task_description
    task = AgentTask(input_path)

    try:
        print(f"Task added: {task_description}")
        result = agent.get_result(task)
        if result is None:
            agent.add_task(task)
            while result is None:
                await asyncio.sleep(0.1)
                result = agent.get_result(task)
        return Response(content=result, media_type="text/plain")

    except Exception as e:
        print(f"Error adding task: {e}")
        return Response(content=f"Error adding task: {e}", status_code=500)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=get_config('server_host', '0.0.0.0'),
        port=int(get_config('server_port', '5000')),
        # reload=True # Optional: Enable for development
    )
