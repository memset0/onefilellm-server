import os
import hashlib
import asyncio
from onefilellm import process_input
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn

dirname = os.path.abspath(os.path.dirname(__file__))


class AgentTask:
    def __init__(self, input_path):
        self.input_path = input_path

    async def run(self):
        print("running task:", self)
        progress = Progress()
        task = progress.add_task("[bright_blue]Processing...", total=None)  # Indeterminate task
        output_file = AgentTask.get_file_path(self)
        output = process_input(self.input_path, progress, task)
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        with open(output_file, "w+", encoding="utf-8") as f:
            f.write(output)
        return output

    @staticmethod
    def get_file_path(task):
        md5_hash = hashlib.md5(task.input_path.encode()).hexdigest()
        return os.path.join(dirname, "cache", f"{md5_hash}.xml")

    def __str__(self):
        return f'task{{{self.input_path}}}'


class Agent:
    def __init__(self):
        self.tasks = []
        self.locked = False
        self.processing_done_event = asyncio.Event()
        self.processing_done_event.set()

        if not os.path.exists(dirname + "/tmp"):
            os.makedirs(dirname + "/tmp")
            os.chdir(dirname + "/tmp")

    def add_task(self, task: AgentTask):
        if self.get_result(task) is not None:
            return

        print("adding new task:", task)
        self.tasks.append(task)
        # Schedule run() to be executed by the event loop
        asyncio.create_task(self.run())

    def get_result(self, task: AgentTask) -> str:
        print("getting result for task:", task)
        
        output_file = AgentTask.get_file_path(task)  # Use static method correctly
        if not os.path.exists(output_file):
            return None
        with open(output_file, "r", encoding="utf-8") as f:  # Added encoding
            result = f.read()
            if len(result) > 0:
                return result
            else:
                return None

    async def run(self):
        if self.locked:  # 说明已经有其他线程在运行run了
            await self.processing_done_event.wait()
            return

        self.locked = True
        self.processing_done_event.clear()

        try:
            while len(self.tasks) > 0:
                task = self.tasks.pop(0)
                await task.run()
        finally:
            self.locked = False
            self.processing_done_event.set()
