from agentdojo.default_suites.v1.shopping.injection_tasks import ShoppingInjectionTask  # noqa: F401 - Register tasks
from agentdojo.default_suites.v1.shopping.task_suite import task_suite as shopping_task_suite
from agentdojo.default_suites.v1.shopping.user_tasks import ShoppingUserTask  # noqa: F401 - Register tasks

__all__ = ["shopping_task_suite"]
