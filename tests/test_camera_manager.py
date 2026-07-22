import unittest

from camera.camera import CameraManager


class CameraManagerTests(unittest.TestCase):
    def test_initialization_with_invalid_camera_does_not_raise(self) -> None:
        manager = CameraManager(source=99999)

        self.assertFalse(manager.available)
        self.assertIsNone(manager.read_frame())


if __name__ == "__main__":
    unittest.main()
