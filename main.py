"""Run the semantic-search API as one Uvicorn worker."""

import logging
import os

import uvicorn

from api.server import application


def main() -> None:
    """Start one worker with one model instance and serialized store writes."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    uvicorn.run(
        application,
        host=os.getenv("APP_HOST", "127.0.0.1"),
        port=int(os.getenv("APP_PORT", "8000")),
        workers=1,
    )


if __name__ == "__main__":
    main()
