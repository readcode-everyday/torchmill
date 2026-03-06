"""Async watcher loop — tails a log file and runs the 4-layer filter stack."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional

from .log_parser import LogParser, MetricsSnapshot
from .signals import Signal, SignalDetector, SignalType

logger = logging.getLogger("torchmill.watcher")


class WatcherLoop:
    """
    Continuously tails a log file, parses lines, detects signals,
    and invokes callbacks for brain triggers and kill triggers.

    Usage::

        loop = WatcherLoop(
            log_path="training.log",
            parser=LogParser(),
            detector=SignalDetector(...),
            check_interval=30,
            on_brain_trigger=my_brain_callback,
            on_kill_trigger=my_kill_callback,
        )
        await loop.run(done_event)
    """

    def __init__(
        self,
        log_path: str,
        parser: LogParser,
        detector: SignalDetector,
        check_interval: float = 30.0,
        on_brain_trigger: Optional[
            Callable[[List[Signal], List[MetricsSnapshot]], Coroutine]
        ] = None,
        on_kill_trigger: Optional[
            Callable[[Signal, List[MetricsSnapshot]], Coroutine]
        ] = None,
    ):
        self.log_path = Path(log_path)
        self.parser = parser
        self.detector = detector
        self.check_interval = check_interval
        self.on_brain_trigger = on_brain_trigger
        self.on_kill_trigger = on_kill_trigger

        self.snapshots: List[MetricsSnapshot] = []
        self._file_offset: int = 0
        self._killed: bool = False

    async def run(self, done_event: asyncio.Event) -> None:
        """
        Run until done_event is set.  done_event should be set when the
        training process exits (or when we receive a kill signal).
        """
        logger.info(f"Watcher started — tailing {self.log_path}")

        while not done_event.is_set():
            await self._tick()
            try:
                await asyncio.wait_for(
                    asyncio.shield(done_event.wait()),
                    timeout=self.check_interval,
                )
            except asyncio.TimeoutError:
                pass

            if self._killed:
                break

        # Final tick to capture any remaining output
        await self._tick()

        # Signal training_completed to brain
        completed = self.detector.training_completed_signal(
            epoch=self._last_epoch()
        )
        if self.on_brain_trigger and not self._killed:
            await self.on_brain_trigger([completed], self.snapshots)

    async def _tick(self) -> None:
        """Read new log lines, parse, detect signals, fire callbacks."""
        new_lines = self._read_new_lines()
        if not new_lines:
            return

        for line in new_lines:
            snap = self.parser.parse_line(line)
            if snap is None:
                continue
            self.snapshots.append(snap)
            signals = self.detector.process(snap)

            for sig in signals:
                if sig.is_kill_trigger:
                    logger.warning(f"Kill trigger: {sig.message}")
                    self._killed = True
                    if self.on_kill_trigger:
                        await self.on_kill_trigger(sig, self.snapshots)
                    return  # stop processing further

            brain_sigs = [s for s in signals if s.is_brain_trigger]
            if brain_sigs and self.on_brain_trigger:
                await self.on_brain_trigger(brain_sigs, self.snapshots)

    def _read_new_lines(self) -> List[str]:
        try:
            with open(self.log_path) as f:
                f.seek(self._file_offset)
                lines = f.readlines()
                self._file_offset = f.tell()
            return lines
        except FileNotFoundError:
            return []

    def _last_epoch(self) -> Optional[int]:
        for snap in reversed(self.snapshots):
            if snap.epoch is not None:
                return snap.epoch
        return None
