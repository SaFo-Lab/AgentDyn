from agentdojo.default_suites.v1.github.injection_tasks import GithubInjectionTask  # noqa: F401 - Register tasks
from agentdojo.default_suites.v1.github.task_suite import task_suite as github_task_suite
from agentdojo.default_suites.v1.github.user_tasks import GithubUserTask  # noqa: F401 - Register tasks

__all__ = ["github_task_suite"]
