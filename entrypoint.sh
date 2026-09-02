#!/bin/sh
# `exec` replaces this shell process with uvicorn, so uvicorn becomes PID 1 and
# receives SIGTERM directly on `docker stop` / platform shutdown, for a clean exit.
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
