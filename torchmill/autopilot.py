"""Autopilot — orchestrates the full run loop (Modes 1 and 2).

Mode 1 (run): spawns training process, watches its output, iterates.
Mode 2 (watch): attaches to an existing log file, optionally can kill PID.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .watcher.log_parser import LogParser, MetricsSnapshot
from .watcher.signals import Signal, SignalDetector
from .watcher.summarizer import Summarizer
from .watcher.loop import WatcherLoop
from .brain.analyzer import BrainAnalyzer
from .history.experiment_log import ExperimentLog
from .watcher_config import TorchmillWatcherConfig

logger = logging.getLogger("torchmill.autopilot")


class Autopilot:
    """Runs the 4-layer filter loop for one or more training rounds."""

    def __init__(self, config: TorchmillWatcherConfig):
        self.cfg = config
        self.brain = BrainAnalyzer(
            provider=config.brain.provider,
            model=config.brain.model,
            api_key=config.brain.api_key,
        )
        self.exp_log = ExperimentLog(log_dir=config.history.dir)
        self._start_time = time.time()

    # ------------------------------------------------------------------ #
    # Public entry points
    # ------------------------------------------------------------------ #

    async def run_autopilot(self) -> None:
        """Mode 1: full autopilot — spawn process, watch, iterate."""
        if not self.cfg.command:
            raise ValueError("torchmill.yaml must specify 'command' for run mode")

        from .actuator.runner import ProcessRunner
        from .actuator.config_mutator import ConfigMutator

        mutator: Optional[ConfigMutator] = None
        if self.cfg.training_config:
            mutator = ConfigMutator(self.cfg.training_config)

        history_base = Path(self.cfg.history.dir)

        for round_num in range(1, self.cfg.autopilot.max_rounds + 1):
            elapsed_hours = (time.time() - self._start_time) / 3600
            if elapsed_hours >= self.cfg.autopilot.max_total_hours:
                logger.info(
                    f"Reached max_total_hours={self.cfg.autopilot.max_total_hours}. "
                    "Stopping."
                )
                break

            logger.info(f"=== Autopilot Round {round_num} ===")

            run_dir = history_base / f"round_{round_num:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            log_file = str(run_dir / "output.log")

            if self.cfg.autopilot.approval_required and round_num > 1:
                _wait_for_approval(round_num)

            runner = ProcessRunner(
                command=self.cfg.command,
                workdir=self.cfg.workdir,
                log_file=log_file,
            )

            current_config: Dict[str, Any] = {}
            if mutator:
                try:
                    current_config = mutator.read()
                except Exception:
                    pass

            action = await self._run_round(
                log_file=log_file,
                round_num=round_num,
                runner=runner,
                current_config=current_config,
            )

            outcome = action.get("action", "continue")
            reason = action.get("reason", "")
            changes = action.get("changes", {})

            # Log round
            self.exp_log.add_round(
                round_num=round_num,
                config=current_config,
                brain_analysis=reason,
                outcome=outcome,
            )

            logger.info(f"Round {round_num} outcome: {outcome} — {reason}")

            if outcome == "kill_and_stop":
                logger.info("Brain says stop. Autopilot done.")
                break
            if outcome == "kill_and_retry" and mutator and changes:
                logger.info(f"Applying hyperparameter changes: {changes}")
                mutator.apply_changes(changes)
            elif outcome == "continue":
                # Training completed normally; check if we should continue
                logger.info("Training completed normally.")
                break

        logger.info("Autopilot finished.")
        print(self.exp_log.summary())

    async def watch_mode(
        self,
        log_file: str,
        external_pid: Optional[int] = None,
    ) -> None:
        """Mode 2: watch an existing log file. Optionally kill external PID."""
        from .actuator.runner import ProcessRunner

        logger.info(
            f"Watch mode: log={log_file}"
            + (f", pid={external_pid}" if external_pid else "")
        )

        done_event = asyncio.Event()
        action: Dict[str, Any] = {"action": "continue", "reason": "", "changes": {}}

        async def on_brain(signals: List[Signal], snaps: List[MetricsSnapshot]):
            nonlocal action
            reason = signals[0].message if signals else ""
            summary = self._build_summary(snaps, round_num=1, trigger=reason)
            result = self.brain.analyze(summary)
            action = result
            logger.info(f"Brain decision: {result['action']} — {result['reason']}")
            if result["action"] in ("kill_and_retry", "kill_and_stop"):
                done_event.set()

        async def on_kill(sig: Signal, snaps: List[MetricsSnapshot]):
            logger.warning(f"Kill trigger fired: {sig.message}")
            if external_pid:
                ProcessRunner.kill_pid(external_pid)
            done_event.set()

        detector = self._make_detector()
        loop = WatcherLoop(
            log_path=log_file,
            parser=LogParser(),
            detector=detector,
            check_interval=self.cfg.watcher.check_interval,
            on_brain_trigger=on_brain,
            on_kill_trigger=on_kill,
        )
        await loop.run(done_event)
        logger.info(f"Watch complete. Final action: {action['action']}")
        print(self.exp_log.summary())

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    async def _run_round(
        self,
        log_file: str,
        round_num: int,
        runner,
        current_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Start process, run watcher loop, return brain action."""
        done_event = asyncio.Event()
        action: Dict[str, Any] = {"action": "continue", "reason": "", "changes": {}}

        async def on_brain(signals: List[Signal], snaps: List[MetricsSnapshot]):
            nonlocal action
            reason = "; ".join(s.message for s in signals)
            summary = self._build_summary(
                snaps,
                round_num=round_num,
                trigger=reason,
                current_config=current_config,
            )
            result = self.brain.analyze(summary)
            action = result
            logger.info(
                f"Brain: {result['action']} — {result['reason']}"
            )
            if result["action"] in ("kill_and_retry", "kill_and_stop"):
                await runner.kill()
                done_event.set()

        async def on_kill(sig: Signal, snaps: List[MetricsSnapshot]):
            logger.warning(f"Kill trigger: {sig.message}")
            await runner.kill()
            done_event.set()

        # Start training process
        await runner.start()

        # Run watcher concurrently with the process
        async def wait_for_process():
            await runner.wait()
            done_event.set()

        detector = self._make_detector()
        watcher = WatcherLoop(
            log_path=log_file,
            parser=LogParser(),
            detector=detector,
            check_interval=self.cfg.watcher.check_interval,
            on_brain_trigger=on_brain,
            on_kill_trigger=on_kill,
        )

        await asyncio.gather(
            watcher.run(done_event),
            wait_for_process(),
        )

        return action

    def _make_detector(self) -> SignalDetector:
        ick = self.cfg.watcher.instant_kill
        btc = self.cfg.watcher.brain_triggers
        return SignalDetector(
            instant_kill_config={
                "nan_loss": ick.nan_loss,
                "grad_norm_above": ick.grad_norm_above,
                "loss_above": ick.loss_above,
            },
            brain_trigger_config={
                "val_loss_rising_for": btc.val_loss_rising_for,
                "loss_plateau_for": btc.loss_plateau_for,
                "every_n_epochs": btc.every_n_epochs,
            },
        )

    def _build_summary(
        self,
        snaps: List[MetricsSnapshot],
        round_num: int = 1,
        trigger: str = "",
        current_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        summ_cfg = self.cfg.watcher.summary
        summarizer = Summarizer(
            max_history_epochs=summ_cfg.max_history_epochs,
            include_experiment_history=summ_cfg.include_experiment_history,
            max_tokens=summ_cfg.max_tokens,
        )
        return summarizer.summarize(
            snapshots=snaps,
            round_num=round_num,
            current_config=current_config,
            experiment_history=self.exp_log.get_rounds(),
            trigger_reason=trigger,
        )


def _wait_for_approval(round_num: int) -> None:
    try:
        answer = input(
            f"\n[torchmill] About to start round {round_num}. "
            "Continue? [Y/n] "
        ).strip().lower()
        if answer in ("n", "no"):
            logger.info("User aborted autopilot.")
            raise SystemExit(0)
    except EOFError:
        pass  # non-interactive; continue
