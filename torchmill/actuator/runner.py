"""Process runner — async subprocess management for training jobs."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import shlex
from pathlib import Path
from typing import Optional

logger = logging.getLogger("torchmill.runner")


class ProcessRunner:
    """Spawn, monitor, and kill a training subprocess."""

    def __init__(
        self,
        command: str,
        workdir: Optional[str] = None,
        log_file: Optional[str] = None,
    ):
        self.command = command
        self.workdir = str(workdir or os.getcwd())
        self.log_file = log_file
        self._process: Optional[asyncio.subprocess.Process] = None
        self._log_fh = None

    @property
    def pid(self) -> Optional[int]:
        return self._process.pid if self._process else None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    @property
    def returncode(self) -> Optional[int]:
        return self._process.returncode if self._process else None

    async def start(self) -> int:
        """Start the process. Returns PID."""
        args = shlex.split(self.command)

        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(log_path, "ab")
            self._process = await asyncio.create_subprocess_exec(
                *args,
                cwd=self.workdir,
                stdout=self._log_fh,
                stderr=self._log_fh,
            )
        else:
            self._process = await asyncio.create_subprocess_exec(
                *args,
                cwd=self.workdir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

        logger.info(f"Started PID={self._process.pid}: {self.command}")
        return self._process.pid

    async def wait(self) -> int:
        """Wait for process completion. Returns exit code."""
        if self._process is None:
            return -1
        code = await self._process.wait()
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None
        return code

    async def kill(self, graceful_timeout: float = 5.0) -> None:
        """SIGTERM → wait → SIGKILL if still alive."""
        if self._process is None or self._process.returncode is not None:
            return
        try:
            self._process.terminate()
            logger.info(f"Sent SIGTERM to PID={self._process.pid}")
            try:
                await asyncio.wait_for(self._process.wait(), timeout=graceful_timeout)
                logger.info(f"Process {self._process.pid} exited cleanly")
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
                logger.warning(f"Sent SIGKILL to PID={self._process.pid}")
        except ProcessLookupError:
            pass  # already gone
        finally:
            if self._log_fh:
                self._log_fh.close()
                self._log_fh = None

    @staticmethod
    def kill_pid(pid: int) -> None:
        """Kill an external process by PID (watch mode)."""
        try:
            os.kill(pid, signal.SIGTERM)
            logger.info(f"Sent SIGTERM to external PID={pid}")
        except ProcessLookupError:
            logger.warning(f"PID {pid} not found")
        except PermissionError:
            logger.error(f"No permission to kill PID {pid}")
