import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required.") from e

from models import PromptguardAction, PromptguardObservation
from server.promptguard_environment import PromptguardEnvironment

DEFAULT_TASK = os.getenv("PROMPTGUARD_TASK", "direct-override")

def make_env():
    """Return a fresh environment instance per session."""
    return PromptguardEnvironment(task_name=DEFAULT_TASK)

app = create_app(
    make_env,
    PromptguardAction,
    PromptguardObservation,
    env_name="promptguard",
    max_concurrent_envs=10,
)

def main(host: str = "0.0.0.0", port: int = 7860):
    """Start the uvicorn server. Call main() to run."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main()
