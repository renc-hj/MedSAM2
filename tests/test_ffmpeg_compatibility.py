import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import app


class FfmpegCompatibilityTest(unittest.TestCase):
    def test_sam2_config_name_is_relative_to_the_package(self):
        config_path = (
            Path(app.__file__).parent
            / "sam2"
            / "configs"
            / "sam2.1_hiera_tiny512_FLARE_RECIST.yaml"
        )

        self.assertEqual(
            app.get_sam2_config_name(config_path),
            "configs/sam2.1_hiera_tiny512_FLARE_RECIST.yaml",
        )

    def test_inference_config_files_exclude_training_configs(self):
        config_dir = Path(app.__file__).parent / "sam2" / "configs"

        config_files = app.get_inference_config_files(config_dir)

        self.assertEqual(
            [path.name for path in config_files],
            ["sam2.1_hiera_t512.yaml"],
        )

    def test_extract_video_frames_works_with_current_ffmpeg(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_video = temp_path / "input.mp4"
            output_dir = temp_path / "frames"
            output_dir.mkdir()

            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=64x64:r=2:d=1",
                    "-pix_fmt",
                    "yuv420p",
                    str(input_video),
                ],
                check=True,
            )

            app.extract_video_frames(
                str(input_video), str(output_dir), frame_interval=1, device="cpu"
            )

            self.assertGreater(len(list(output_dir.glob("*.jpg"))), 0)

    def test_extract_video_frames_falls_back_when_cuda_decode_is_unavailable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_video = temp_path / "input.mp4"
            output_dir = temp_path / "frames"
            output_dir.mkdir()

            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=64x64:r=2:d=1",
                    "-pix_fmt",
                    "yuv420p",
                    str(input_video),
                ],
                check=True,
            )

            real_run = app._run_ffmpeg_frame_extraction
            input_args_seen = []

            def fail_cuda_then_run_on_cpu(*args, input_args):
                input_args_seen.append(input_args)
                if input_args == {"hwaccel": "cuda"}:
                    raise app.ffmpeg.Error("ffmpeg", b"", b"CUDA unavailable")
                return real_run(*args, input_args=input_args)

            with mock.patch.object(
                app,
                "_run_ffmpeg_frame_extraction",
                side_effect=fail_cuda_then_run_on_cpu,
            ):
                app.extract_video_frames(
                    str(input_video), str(output_dir), frame_interval=1, device="cuda"
                )

            self.assertGreater(len(list(output_dir.glob("*.jpg"))), 0)
            self.assertEqual(input_args_seen, [{"hwaccel": "cuda"}, {}])


if __name__ == "__main__":
    unittest.main()
