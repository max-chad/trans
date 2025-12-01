import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

# Ensure we can import from app
sys.path.append(str(Path(__file__).parent))

from PyQt6.QtWidgets import QApplication
from app.config import AppConfig
from ui.main_window import MainWindow

# Create a single QApplication instance for all tests
app = QApplication(sys.argv)

class TestMainWindowLogic(unittest.TestCase):
    def setUp(self):
        self.config = AppConfig(Path("dummy_config.json"))
        # Mock config to avoid file I/O
        self.config.save_config = MagicMock()
        self.config.load_config = MagicMock(return_value=self.config.defaults)
        
        # Mock dependencies that might require GPU or heavy loading
        with patch("ui.main_window.detect_supported_nvidia_gpus") as mock_gpu, \
             patch("ui.main_window.TranscriptionWorker"), \
             patch("ui.main_window.TranslationWorker"):
            
            mock_gpu.return_value = MagicMock(torch_usable=False, has_supported_series=False, any_gpu=False)
            self.window = MainWindow(self.config)

    def tearDown(self):
        self.window.close()

    def test_clear_all_tasks(self):
        # Add a dummy task
        task_id = "task_1"
        mock_task = MagicMock()
        mock_task.status = "queued"
        mock_task.task_id = task_id
        
        self.window.tasks[task_id] = mock_task
        mock_widget = MagicMock()
        self.window.task_widgets[task_id] = mock_widget
        
        # Verify task is added
        self.assertIn(task_id, self.window.tasks)
        
        # Clear tasks
        self.window.clear_all_tasks()
        
        # Verify tasks are removed
        self.assertNotIn(task_id, self.window.tasks)
        self.assertNotIn(task_id, self.window.task_widgets)
        mock_widget.deleteLater.assert_called()

    def test_output_mode_switch(self):
        # Default mode should be 'source' (from defaults)
        self.assertEqual(self.window.output_mode_combo.currentData(), "source")
        self.assertFalse(self.window.output_label.isVisible())
        self.assertFalse(self.window.output_btn.isVisible())
        
        # Switch to 'custom'
        idx = self.window.output_mode_combo.findData("custom")
        self.window.output_mode_combo.setCurrentIndex(idx)
        
        self.assertEqual(self.window.output_mode_combo.currentData(), "custom")
        self.assertTrue(self.window.output_label.isVisible())
        self.assertTrue(self.window.output_btn.isVisible())

    @patch("ui.main_window.TranscriptionTask")
    def test_add_video_files_source_mode(self, MockTask):
        # Setup source mode
        self.window.config.set("output_mode", "source")
        
        video_path = Path("/tmp/video.mp4")
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "is_file", return_value=True), \
             patch("ui.main_window.MainWindow._is_supported_media", return_value=True):
            
            self.window.add_video_files([video_path])
            
            # Check that task was created with correct output_dir
            call_args = MockTask.call_args
            self.assertIsNotNone(call_args)
            _, kwargs = call_args
            self.assertEqual(kwargs["output_dir"], video_path.parent)

    @patch("ui.main_window.TranscriptionTask")
    def test_add_video_files_custom_mode(self, MockTask):
        # Setup custom mode
        custom_dir = Path("/custom/output")
        self.window.config.set("output_mode", "custom")
        self.window.config.set("output_dir", str(custom_dir))
        
        video_path = Path("/tmp/video.mp4")
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "is_file", return_value=True), \
             patch("ui.main_window.MainWindow._is_supported_media", return_value=True):
            
            self.window.add_video_files([video_path])
            
            # Check that task was created with correct output_dir
            call_args = MockTask.call_args
            self.assertIsNotNone(call_args)
            _, kwargs = call_args
            self.assertEqual(kwargs["output_dir"], Path(str(custom_dir)))

if __name__ == "__main__":
    unittest.main()
