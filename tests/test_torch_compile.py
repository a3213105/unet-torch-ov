import argparse
import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch._inductor.config as inductor_cfg

import unet_benchmark as benchmark


class TorchCompileTests(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Conv2d(3, 2, 1).eval()
        self.spec = benchmark.ModelSpec(
            "scale0.5", 0.5, Path("model.pth"), Path("model.xml"), Path("model_all.xml")
        )
        self.args = argparse.Namespace(
            skip_pytorch=False,
            skip_ov=True,
            skip_all=True,
            torch_compile=1,
            images=[Path("image.png")],
            batch_size=1,
            precisions=["FP32", "BF16", "FP16"],
        )

    def run_benchmark(self, expected_failures=0, expected_results=3):
        failures = []
        with patch.object(
            benchmark,
            "preprocess_image",
            return_value=(np.zeros((1, 3, 4, 4), dtype=np.float32), (4, 4)),
        ):
            results = benchmark.benchmark_dataset(
                self.spec, self.model, self.args, 1, {}, {}, 1, 1, failures
            )
        self.assertEqual(len(failures), expected_failures)
        self.assertEqual(len(results), expected_results)
        return results

    def test_compile_enables_persistent_freezing_and_prepacking(self):
        self.model.train()

        def compile_model(model, **kwargs):
            self.assertIs(model, self.model)
            self.assertFalse(model.training)
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(inductor_cfg.freezing)
            self.assertTrue(inductor_cfg.cpp.weight_prepack)
            self.assertEqual(kwargs, {"backend": "inductor"})
            return model

        def check_execution_config(model, inputs):
            self.assertTrue(inductor_cfg.freezing)
            self.assertTrue(inductor_cfg.cpp.weight_prepack)
            self.assertFalse(torch.is_grad_enabled())

        handle = self.model.register_forward_pre_hook(check_execution_config)
        try:
            with inductor_cfg.patch({"freezing": False, "cpp.weight_prepack": False}):
                with patch.object(torch, "compile", side_effect=compile_model) as compile_mock:
                    results = self.run_benchmark()
                self.assertFalse(inductor_cfg.freezing)
                self.assertFalse(inductor_cfg.cpp.weight_prepack)
        finally:
            handle.remove()
        self.assertEqual(compile_mock.call_count, 3)
        for result in results:
            self.assertEqual(result["torch_compile_backend"], "inductor")
            self.assertTrue(result["torch_inductor_freezing"])
            self.assertTrue(result["torch_inductor_weight_prepack"])
            self.assertEqual(result["torch_compile_error"], "")

    def test_eager_does_not_compile(self):
        self.args.torch_compile = 0
        with patch.object(torch, "compile") as compile_mock:
            results = self.run_benchmark()
        compile_mock.assert_not_called()
        for result in results:
            self.assertEqual(result["torch_compile_backend"], "eager")
            self.assertFalse(result["torch_inductor_freezing"])
            self.assertFalse(result["torch_inductor_weight_prepack"])
            self.assertEqual(result["torch_compile_seconds"], 0.0)

    def test_compile_setup_failure_reports_eager_fallback(self):
        self.check_fallback(RuntimeError("compile setup failed"))

    def test_compile_first_execution_failure_reports_eager_fallback(self):
        def failed_model(inputs):
            raise RuntimeError("first execution failed")

        self.check_fallback(None, failed_model)

    def test_timed_execution_failure_restores_config(self):
        with inductor_cfg.patch({"freezing": False, "cpp.weight_prepack": False}):
            with contextlib.redirect_stdout(io.StringIO()), patch.object(
                torch, "compile", return_value=self.model
            ), patch.object(
                benchmark, "timed_iterations", side_effect=RuntimeError("timed execution failed")
            ):
                self.run_benchmark(expected_failures=3, expected_results=0)
            self.assertFalse(inductor_cfg.freezing)
            self.assertFalse(inductor_cfg.cpp.weight_prepack)

    def check_fallback(self, error, compiled_model=None):
        output = io.StringIO()

        def check_eager_config(model, inputs):
            self.assertFalse(inductor_cfg.freezing)
            self.assertFalse(inductor_cfg.cpp.weight_prepack)

        handle = self.model.register_forward_pre_hook(check_eager_config)
        try:
            with inductor_cfg.patch({"freezing": False, "cpp.weight_prepack": False}):
                with contextlib.redirect_stdout(output), patch.object(
                    torch, "compile", side_effect=error, return_value=compiled_model
                ):
                    results = self.run_benchmark()
                self.assertFalse(inductor_cfg.freezing)
                self.assertFalse(inductor_cfg.cpp.weight_prepack)
        finally:
            handle.remove()
        self.assertEqual(output.getvalue().count("falling back to eager"), 3)
        for result in results:
            self.assertEqual(result["torch_compile_backend"], "eager")
            self.assertFalse(result["torch_inductor_freezing"])
            self.assertFalse(result["torch_inductor_weight_prepack"])
            self.assertIn("RuntimeError", result["torch_compile_error"])


if __name__ == "__main__":
    unittest.main()
