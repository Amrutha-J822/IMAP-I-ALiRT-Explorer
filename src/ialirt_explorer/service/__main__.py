"""Entrypoint: ``python -m ialirt_explorer.service`` (or via the script alias)."""

from __future__ import annotations

import logging
import os

import uvicorn


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        level=os.environ.get("IALIRT_LOG_LEVEL", "INFO").upper(),
        datefmt="%H:%M:%S",
    )
    host = os.environ.get("IALIRT_SERVICE_HOST", "0.0.0.0")
    # PORT is the platform-as-a-service convention (Render, Fly, Heroku, Cloud Run).
    port = int(
        os.environ.get("PORT")
        or os.environ.get("IALIRT_SERVICE_PORT")
        or "8000"
    )

    uvicorn.run(
        "ialirt_explorer.service.api:app",
        host=host,
        port=port,
        reload=os.environ.get("IALIRT_SERVICE_RELOAD", "0") == "1",
    )


if __name__ == "__main__":
    main()
