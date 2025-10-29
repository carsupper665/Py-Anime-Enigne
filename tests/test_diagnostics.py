import json
import logging
from pathlib import Path

from core.diagnostics import attach_diagnostic, format_user_message, generate_diagnostic_id, log_structured
from core.export_context import ExportOptions
from core.export.services import ExportCommandBuilder, TempDirectoryManager


def test_attach_diagnostic_sets_attribute():
    exc = ValueError("boom")
    diag = generate_diagnostic_id()
    result = attach_diagnostic(exc, diag)
    assert getattr(result, "diagnostic_id") == diag


def test_format_user_message_appends_id():
    diag = "diag-test"
    message = format_user_message("Error occurred", diag)
    assert diag in message


def test_temp_directory_manager_logs_copy(tmp_path, caplog):
    logger = logging.getLogger("test.tempdir")
    manager = TempDirectoryManager(base_dir=str(tmp_path), logger=logger)
    src = tmp_path / "source.txt"
    src.write_text("payload", encoding="utf-8")
    diag = generate_diagnostic_id()

    with caplog.at_level(logging.INFO, logger="test.tempdir"):
        dest = manager.copy_to_temp(str(src), diagnostic_id=diag)

    record = caplog.records[0]
    payload = json.loads(record.message)
    assert payload["diagnostic_id"] == diag
    assert payload["event"] == "temp.copy"
    assert Path(payload["target"]).exists()
    assert payload["target"] == dest


def test_export_command_builder_logs(caplog):
    logger = logging.getLogger("test.builder")
    builder = ExportCommandBuilder(logger=logger)
    options = ExportOptions()
    diag = generate_diagnostic_id()

    with caplog.at_level(logging.INFO, logger="test.builder"):
        cmd = builder.build_trim_command("input.mp4", 0, 1000, "output.mp4", diagnostic_id=diag)

    assert cmd[0] == "ffmpeg"
    payload = json.loads(caplog.records[0].message)
    assert payload["diagnostic_id"] == diag
    assert payload["event"] == "command.build.trim"
    assert payload["src"] == "input.mp4"
