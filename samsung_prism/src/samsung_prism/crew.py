from crewai import Agent, Crew, Task, Process, LLM
from crewai.project import CrewBase, agent, task, crew
import yaml
from pathlib import Path


BASE_DIR = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "config"





@CrewBase
class SamsungCompetitorIntelligenceCrew:
    """Crew 1 - Samsung Competitor Intelligence"""

    def __init__(self):
        with open(CONFIG_DIR / "agents.yaml", "r") as f:
            self.agents_config = yaml.safe_load(f)

        with open(CONFIG_DIR / "tasks.yaml", "r") as f:
            self.tasks_config = yaml.safe_load(f)

    # ------------------ AGENTS ------------------

    @agent
    def web_recon_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["web_recon_agent"],
            llm=LLM(model="groq/llama-3.3-70b-versatile", temperature=0.4),
            reasoning=False,
            max_iter=10,
            allow_delegation=False,
        )

    @agent
    def social_spy_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["social_spy_agent"],
            llm=LLM(model="groq/llama-3.3-70b-versatile", temperature=0.4),
            reasoning=False,
            max_iter=10,
            allow_delegation=False,
        )

    @agent
    def hiring_talent_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["hiring_talent_agent"],
            llm=LLM(model="groq/llama-3.3-70b-versatile", temperature=0.3),
            reasoning=False,
            max_iter=10,
            allow_delegation=False,
        )

    @agent
    def patent_rd_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["patent_rd_agent"],
            llm=LLM(model="groq/llama-3.3-70b-versatile", temperature=0.3),
            reasoning=False,
            max_iter=10,
            allow_delegation=False,
        )

    @agent
    def pricing_tracker_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["pricing_tracker_agent"],
            llm=LLM(model="groq/llama-3.3-70b-versatile", temperature=0.3),
            reasoning=False,
            max_iter=10,
            allow_delegation=False,
        )

    # ------------------ TASKS ------------------

    @task
    def web_recon_task(self) -> Task:
        return Task(
            config=self.tasks_config["web_recon_task"],
            agent=self.web_recon_agent(),
        )

    @task
    def social_spy_task(self) -> Task:
        return Task(
            config=self.tasks_config["social_spy_task"],
            agent=self.social_spy_agent(),
        )

    @task
    def hiring_talent_task(self) -> Task:
        return Task(
            config=self.tasks_config["hiring_talent_task"],
            agent=self.hiring_talent_agent(),
        )

    @task
    def patent_rd_task(self) -> Task:
        return Task(
            config=self.tasks_config["patent_rd_task"],
            agent=self.patent_rd_agent(),
        )

    @task
    def pricing_tracker_task(self) -> Task:
        return Task(
            config=self.tasks_config["pricing_tracker_task"],
            agent=self.pricing_tracker_agent(),
        )

    # ------------------ CREW ------------------

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            task_execution_delay=6  # seconds
        )

