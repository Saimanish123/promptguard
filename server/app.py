import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Install with: pip install openenv-core") from e

from models import PromptguardAction, PromptguardObservation
from server.promptguard_environment import PromptguardEnvironment

TASK_NAME = os.getenv("PROMPTGUARD_TASK", "direct-override")

_env_instance = PromptguardEnvironment(task_name=TASK_NAME)


def make_env():
    """Return the singleton environment instance."""
    return _env_instance


app = create_app(
    make_env,
    PromptguardAction,
    PromptguardObservation,
    env_name="promptguard",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Start the uvicorn server. Call main() to run."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main()
