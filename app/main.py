import argparse

from app.app.commands import (
    AnalysisReportCommand,
    CameraCheckCommand,
    DoctorCommand,
    DownloadModelsCommand,
    UiCommand,
)
from app.app.handlers import (
    handle_analysis_report,
    handle_camera_check,
    handle_doctor,
    handle_download_models,
    handle_ui,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="face-recognition-app",
        description="Desktop face recognition app built with OpenCV DNN and uv.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("ui", help="Launch the desktop application.")
    subparsers.add_parser("doctor", help="Validate dependencies and model files.")
    subparsers.add_parser(
        "download-models",
        help="Download the official YuNet, SFace, and Face Landmarker models into ./models.",
    )
    analysis_parser = subparsers.add_parser(
        "analysis-report",
        help="Generate an HTML analysis report from SQLite data.",
    )
    analysis_parser.add_argument(
        "--open",
        action="store_true",
        dest="open_browser",
        help="Open the generated report in the default browser.",
    )

    camera_parser = subparsers.add_parser(
        "camera-check",
        help="Open the default camera and capture a single frame.",
    )
    camera_parser.add_argument("--camera-index", type=int, default=0)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command in (None, "ui"):
        return handle_ui(UiCommand())
    if args.command == "doctor":
        return handle_doctor(DoctorCommand())
    if args.command == "download-models":
        return handle_download_models(DownloadModelsCommand())
    if args.command == "analysis-report":
        return handle_analysis_report(
            AnalysisReportCommand(open_browser=args.open_browser)
        )
    if args.command == "camera-check":
        return handle_camera_check(CameraCheckCommand(camera_index=args.camera_index))

    parser.print_help()
    return 1
