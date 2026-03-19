"""
Scene Editing Agent Module
LLM-driven autonomous 3D scene editing agent using Gemini Function Calling.
Implements a ReAct-style agent loop that reasons about user intent and
executes multi-step editing operations with visual feedback.
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import torch
import numpy as np
from PIL import Image

from .metrics import SemanticStateEvaluator, SemanticEvalResult

logger = logging.getLogger(__name__)


class SceneAgent:
    """
    LLM-driven autonomous 3D scene editing agent.

    This agent wraps the SceneEditor with an LLM-powered reasoning loop.
    Users interact via natural language; the LLM determines which tools
    to call, observes results, and iterates until the task is complete.

    Architecture:
        User message → LLM reasoning → Tool call → Execution → Observation
        → LLM reasoning → ... → Final text response

    Attributes:
        system: SceneLangSystem instance (trained scene representation).
        editor: SceneEditor instance (low-level editing operations).
        llm: SceneAgentLLM instance (LLM with Function Calling support).
        edit_history: Stack of editing states for undo support.
        max_iterations: Maximum tool-call iterations per user message.
        output_dir: Directory for saving rendered images.
    """

    # Maximum number of tool-call iterations per user message
    DEFAULT_MAX_ITERATIONS = 10

    def __init__(
        self,
        system,
        editor,
        llm,
        device: str = "cuda",
        output_dir: Optional[str] = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ):
        """
        Initialize the SceneAgent.

        Args:
            system: Trained SceneLangSystem instance.
            editor: SceneEditor instance bound to the system.
            llm: SceneAgentLLM instance with Function Calling enabled.
            device: Compute device ('cuda' or 'cpu').
            output_dir: Directory for saving rendered images.
                        If None, uses a temporary directory.
            max_iterations: Maximum tool-call iterations per user message.
        """
        self.system = system
        self.editor = editor
        self.llm = llm
        self.device = device
        self.max_iterations = max_iterations
        self.edit_history: List[Dict[str, Any]] = []

        # Setup output directory
        if output_dir is not None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._tmp_dir = tempfile.TemporaryDirectory()
            self.output_dir = Path(self._tmp_dir.name)

        # Render counter for unique filenames
        self._render_counter = 0

        # Semantic State Evaluator (Direction A: closed-loop feedback)
        self.evaluator = SemanticStateEvaluator(system, device=device)
        self._last_eval: Optional[SemanticEvalResult] = None
        self._last_edit_mask: Optional[torch.Tensor] = None
        self._pre_edit_xyz: Optional[torch.Tensor] = None

        # Tool dispatch table
        self._tool_dispatch = {
            "select_and_highlight": self._tool_select_and_highlight,
            "translate_object": self._tool_translate_object,
            "rotate_object": self._tool_rotate_object,
            "scale_object": self._tool_scale_object,
            "delete_object": self._tool_delete_object,
            "describe_scene": self._tool_describe_scene,
            "undo_last_edit": self._tool_undo_last_edit,
            "render_current_scene": self._tool_render_current_scene,
            "evaluate_semantic_consistency": self._tool_evaluate_semantic_consistency,
            "adaptive_finetune": self._tool_adaptive_finetune,
        }

        logger.info(
            "[SceneAgent] Initialized with %d available tools, "
            "max_iterations=%d, output_dir=%s",
            len(self._tool_dispatch),
            self.max_iterations,
            self.output_dir,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """
        Main dialogue entry point.

        Sends the user message to the LLM along with current scene images.
        The LLM may invoke tools (Function Calling) in a loop until it
        produces a final text response.

        Args:
            user_message: Natural-language instruction or question.

        Returns:
            Final text response from the LLM.
        """
        logger.info("[SceneAgent] User: %s", user_message)

        # Render current scene as visual context for the LLM
        rgb_path, pca_path = self._render_scene_pair()

        # First turn: send user message + scene images
        response = self.llm.send_message(
            text=user_message,
            image_paths=[str(rgb_path), str(pca_path)],
        )

        # Agent loop: process tool calls until LLM returns text
        for iteration in range(self.max_iterations):
            # Check if the response contains function call(s)
            function_calls = self._extract_function_calls(response)
            if function_calls is None:
                # No function call → LLM returned a final text response
                final_text = self._extract_text(response)
                logger.info("[SceneAgent] Response: %s", final_text[:200])
                return final_text

            # Process all function calls (Gemini may return parallel calls)
            all_results = []
            for tool_name, tool_args in function_calls:
                logger.info(
                    "[SceneAgent] Iteration %d: calling tool '%s' with args %s",
                    iteration + 1,
                    tool_name,
                    tool_args,
                )

                # Execute the tool
                tool_result = self._execute_tool(tool_name, tool_args)
                logger.info(
                    "[SceneAgent] Tool '%s' result: %s",
                    tool_name,
                    {k: v for k, v in tool_result.items() if k != "mask"},
                )
                all_results.append((tool_name, tool_result))

            # Send all tool results back to LLM
            # For parallel calls, send results as a single message
            if len(all_results) == 1:
                response = self.llm.send_tool_result(
                    all_results[0][0], all_results[0][1]
                )
            else:
                # Multiple results: send as combined FunctionResponse parts
                response = self._send_multiple_tool_results(all_results)

        # Safety: reached maximum iterations
        logger.warning(
            "[SceneAgent] Reached max_iterations=%d without final response",
            self.max_iterations,
        )
        return "I have reached the maximum number of steps. Please try a simpler request."

    def reset_history(self):
        """Clear the editing history stack."""
        self.edit_history.clear()
        logger.info("[SceneAgent] Edit history cleared")

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dispatch a tool call to the corresponding handler method.

        Args:
            tool_name: Name of the tool to execute.
            tool_args: Arguments for the tool.

        Returns:
            Dict containing the tool execution result.
        """
        handler = self._tool_dispatch.get(tool_name)
        if handler is None:
            logger.error("[SceneAgent] Unknown tool: %s", tool_name)
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

        try:
            return handler(**tool_args)
        except Exception as e:
            logger.exception("[SceneAgent] Tool '%s' raised an exception", tool_name)
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _tool_select_and_highlight(
        self,
        object_description: str,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Select an object by language description and return statistics.

        Args:
            object_description: Text description of the target object.
            threshold: Similarity threshold for CLIP-based selection.

        Returns:
            Dict with selection statistics (count, center, bounding box).
        """
        mask = self.editor.select_object_by_prompt(object_description, threshold)
        num_selected = mask.sum().item()

        if num_selected == 0:
            return {
                "success": False,
                "message": f"No object matching '{object_description}' found. "
                           f"Try lowering the threshold (current: {threshold}).",
                "num_selected": 0,
            }

        # Compute spatial statistics of selected Gaussians
        positions = self.system.geometry.get_xyz[mask]  # [M, 3]
        center = positions.mean(dim=0).cpu().tolist()
        bbox_min = positions.min(dim=0).values.cpu().tolist()
        bbox_max = positions.max(dim=0).values.cpu().tolist()

        return {
            "success": True,
            "num_selected": num_selected,
            "total_gaussians": len(mask),
            "center": center,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
        }

    def _tool_translate_object(
        self,
        object_description: str,
        dx: float,
        dy: float,
        dz: float,
    ) -> Dict[str, Any]:
        """
        Translate an object and return result with rendered image.

        Args:
            object_description: Text description of the target object.
            dx: Translation along x-axis.
            dy: Translation along y-axis.
            dz: Translation along z-axis.

        Returns:
            Dict with editing result and rendered image path.
        """
        self._pre_edit_save()

        result = self.editor.edit_scene(
            prompt=object_description,
            operation="translate",
            offset=(dx, dy, dz),
            semantic_update=False,  # Agent decides via evaluate + finetune
        )

        if result["success"]:
            self._post_edit_update(result, "translate", object_description)
            result["rendered_image"] = str(self._render_scene_pair()[0])

        return self._sanitize_result(result)

    def _tool_rotate_object(
        self,
        object_description: str,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> Dict[str, Any]:
        """
        Rotate an object and return result with rendered image.

        Args:
            object_description: Text description of the target object.
            roll: Rotation around x-axis in degrees.
            pitch: Rotation around y-axis in degrees.
            yaw: Rotation around z-axis in degrees.

        Returns:
            Dict with editing result and rendered image path.
        """
        self._pre_edit_save()

        result = self.editor.edit_scene(
            prompt=object_description,
            operation="rotate",
            rotation=(roll, pitch, yaw),
            semantic_update=False,  # Agent decides via evaluate + finetune
        )

        if result["success"]:
            self._post_edit_update(result, "rotate", object_description)
            result["rendered_image"] = str(self._render_scene_pair()[0])

        return self._sanitize_result(result)

    def _tool_scale_object(
        self,
        object_description: str,
        scale_factor: float,
    ) -> Dict[str, Any]:
        """
        Scale an object and return result with rendered image.

        Args:
            object_description: Text description of the target object.
            scale_factor: Scaling factor (1.0 = no change).

        Returns:
            Dict with editing result and rendered image path.
        """
        if scale_factor <= 0:
            return {"success": False, "error": "scale_factor must be positive"}

        self._pre_edit_save()

        result = self.editor.edit_scene(
            prompt=object_description,
            operation="scale",
            scale_factor=scale_factor,
            semantic_update=False,  # Agent decides via evaluate + finetune
        )

        if result["success"]:
            self._post_edit_update(result, "scale", object_description)
            result["rendered_image"] = str(self._render_scene_pair()[0])

        return self._sanitize_result(result)

    def _tool_delete_object(
        self,
        object_description: str,
        inpaint_background: bool = True,
    ) -> Dict[str, Any]:
        """
        Delete an object and optionally inpaint the background.

        Args:
            object_description: Text description of the object to delete.
            inpaint_background: Whether to inpaint the background.

        Returns:
            Dict with editing result and rendered image path.
        """
        self._pre_edit_save()

        result = self.editor.edit_scene(
            prompt=object_description,
            operation="delete",
            inpaint_background=inpaint_background,
            semantic_update=False,  # Agent decides via evaluate + finetune
        )

        if result["success"]:
            self._post_edit_update(result, "delete", object_description)
            result["rendered_image"] = str(self._render_scene_pair()[0])

        return self._sanitize_result(result)

    def _tool_describe_scene(self, question: str) -> Dict[str, Any]:
        """
        Answer a question about the scene using LLM + visual context.

        Args:
            question: Question about the scene content.

        Returns:
            Dict with the answer text.
        """
        rgb_path, pca_path = self._render_scene_pair()
        answer = self.llm.chat_with_scene(
            query=question,
            image_paths=[str(rgb_path), str(pca_path)],
        )
        return {"success": True, "answer": answer}

    def _tool_undo_last_edit(self) -> Dict[str, Any]:
        """
        Undo the most recent editing operation.

        Returns:
            Dict with undo result.
        """
        if not self.edit_history:
            return {"success": False, "message": "No edit to undo."}

        last_edit = self.edit_history.pop()
        self.editor.restore_parameters()

        logger.info(
            "[SceneAgent] Undone: %s on '%s'",
            last_edit["operation"],
            last_edit["target"],
        )
        return {
            "success": True,
            "message": f"Undone: {last_edit['operation']} on '{last_edit['target']}'",
            "rendered_image": str(self._render_scene_pair()[0]),
        }

    def _tool_render_current_scene(
        self,
        description: str = "",
    ) -> Dict[str, Any]:
        """
        Render the current scene state and return image paths.

        Args:
            description: Optional description for logging.

        Returns:
            Dict with rendered image paths.
        """
        if description:
            logger.info("[SceneAgent] Rendering scene: %s", description)

        rgb_path, pca_path = self._render_scene_pair()
        return {
            "success": True,
            "rgb_image": str(rgb_path),
            "semantic_image": str(pca_path),
        }

    # ------------------------------------------------------------------
    # Semantic evaluation tools (Direction A: closed-loop feedback)
    # ------------------------------------------------------------------

    def _tool_evaluate_semantic_consistency(self) -> Dict[str, Any]:
        """
        Evaluate semantic consistency of the scene after an edit.

        This is the core closed-loop mechanism: the LLM observes
        quantitative metrics and decides whether to finetune, undo,
        or proceed.

        Returns:
            Dict with all semantic evaluation metrics.
        """
        logger.info("[SceneAgent] Evaluating semantic consistency...")

        eval_result = self.evaluator.evaluate_post_edit(
            edit_mask=self._last_edit_mask,
            selection_mask=self._last_edit_mask,
            relevancy_scores=self._last_relevancy,
        )
        self._last_eval = eval_result

        return {
            "success": True,
            **eval_result.to_dict(),
            "interpretation": self._interpret_eval(eval_result),
        }

    def _tool_adaptive_finetune(
        self,
        override_steps: Optional[int] = None,
        learning_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run adaptive semantic fine-tuning based on evaluation metrics.

        The number of steps is determined by the last evaluation result
        unless explicitly overridden.

        Args:
            override_steps: Override the auto-computed step count.
            learning_rate: Override the default learning rate.

        Returns:
            Dict with finetune result and updated metrics.
        """
        if self._last_edit_mask is None:
            return {
                "success": False,
                "error": "No recent edit found. Run an edit first.",
            }

        # Determine steps
        if override_steps is not None:
            num_steps = max(1, min(override_steps, 500))
        elif self._last_eval is not None:
            num_steps = self._last_eval.recommended_finetune_steps
        else:
            num_steps = 30

        lr = learning_rate if learning_rate is not None else 0.005

        logger.info(
            "[SceneAgent] Running adaptive finetune: %d steps, lr=%.4f",
            num_steps, lr,
        )

        # Save pre-finetune SCS for improvement tracking
        self._pre_finetune_scs = (
            self._last_eval.semantic_consistency_score
            if self._last_eval is not None
            else None
        )

        # NOTE: Do NOT call snapshot_pre_edit() here — it would overwrite the
        # original pre-edit baseline used by evaluate_semantic_consistency.
        # The finetune improvement is tracked via _pre_finetune_scs instead.

        # Run localized semantic finetune
        if hasattr(self.system, "local_semantic_finetune"):
            self.system.local_semantic_finetune(
                affected_mask=self._last_edit_mask,
                num_steps=num_steps,
                lr=lr,
            )

        # Re-evaluate after finetune
        post_eval = self.evaluator.evaluate_post_edit(
            edit_mask=self._last_edit_mask,
        )
        self._last_eval = post_eval

        # Compute improvement relative to pre-finetune state
        improvement = ""
        if self._pre_finetune_scs is not None:
            delta = post_eval.semantic_consistency_score - self._pre_finetune_scs
            improvement = f"SCS improved by {delta:+.3f}"

        return {
            "success": True,
            "steps_executed": num_steps,
            "learning_rate": lr,
            **post_eval.to_dict(),
            "improvement": improvement,
            "interpretation": self._interpret_eval(post_eval),
        }

    @staticmethod
    def _interpret_eval(eval_result: SemanticEvalResult) -> str:
        """
        Generate a human-readable interpretation of metrics for the LLM.

        Args:
            eval_result: Semantic evaluation result.

        Returns:
            Interpretation string the LLM can use for reasoning.
        """
        parts = []

        scs = eval_result.semantic_consistency_score
        if scs >= 0.85:
            parts.append("Semantic consistency is EXCELLENT.")
        elif scs >= 0.7:
            parts.append("Semantic consistency is ACCEPTABLE.")
        elif scs >= 0.5:
            parts.append(
                "Semantic consistency is LOW. Consider adaptive_finetune."
            )
        else:
            parts.append(
                "Semantic consistency is POOR. Recommend adaptive_finetune "
                "or undo_last_edit."
            )

        fds = eval_result.feature_distribution_shift
        if fds > 0.5:
            parts.append(
                f"Feature drift is HIGH ({fds:.2f}). Editing may have "
                "introduced artifacts."
            )

        urp = eval_result.unedited_preservation_score
        if urp < 0.8:
            parts.append(
                f"Unedited regions were disturbed (URP={urp:.2f}). "
                "This suggests the edit affected nearby objects."
            )

        mvc = eval_result.multi_view_consistency
        if mvc < 0.8:
            parts.append(
                f"Multi-view consistency is LOW (MVC={mvc:.2f}). "
                f"Worst view SCS={eval_result.worst_view_scs:.2f}. "
                "Some viewpoints are significantly worse than others."
            )

        steps = eval_result.recommended_finetune_steps
        if steps > 50:
            parts.append(
                f"Recommended {steps} finetune steps (elevated — "
                "significant semantic repair needed)."
            )

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Pre/post-edit processing
    # ------------------------------------------------------------------

    def _pre_edit_save(self):
        """
        Snapshot state before an edit for evaluation and target warping.

        Saves:
            - Pre-edit rendered views for SCS/URP computation.
            - Undo backup on the parameter stack.
            - Pre-edit 3D positions for Direction B target warping.
        """
        self.evaluator.snapshot_pre_edit(num_views=2)
        self.editor.backup_parameters()
        self._pre_edit_xyz = self.system.geometry.get_xyz.data.clone()

    def _post_edit_update(
        self,
        result: Dict[str, Any],
        operation: str,
        target: str,
    ):
        """
        Run post-edit processing: record history and cache edit mask.

        Semantic finetuning is NO LONGER triggered automatically here.
        Instead, the LLM agent decides whether to call adaptive_finetune
        based on the semantic evaluation metrics (Direction A).

        Args:
            result: Edit result dict (must contain 'mask' key).
            operation: Operation name for the edit history.
            target: Target object description for the edit history.
        """
        # Record in edit history
        self.edit_history.append({
            "operation": operation,
            "target": target,
            "num_selected": result.get("num_selected", 0),
        })

        # Cache edit mask and selection metadata for evaluation
        self._last_edit_mask = result.get("mask")
        # Cache relevancy/uncertainty from the editor's last selection
        self._last_relevancy = getattr(self.editor, "_last_relevancy", None)
        self._last_uncertainty = getattr(self.editor, "_last_uncertainty", None)

        # Direction B: Warp semantic targets for geometric edits
        mask = result.get("mask")
        if (
            mask is not None
            and operation in ("translate", "rotate", "scale")
            and self._pre_edit_xyz is not None
            and hasattr(self.system, "warp_semantic_targets")
            and mask.shape[0] == self._pre_edit_xyz.shape[0]
        ):
            pre_xyz = self._pre_edit_xyz[mask]
            post_xyz = self.system.geometry.get_xyz.data[mask].clone()
            if not torch.allclose(pre_xyz, post_xyz, atol=1e-6):
                self.system.warp_semantic_targets(pre_xyz, post_xyz)
        self._pre_edit_xyz = None

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _render_scene_pair(self) -> Tuple[Path, Path]:
        """
        Render the current scene as RGB + PCA semantic images.

        Returns:
            Tuple of (rgb_image_path, pca_image_path).
        """
        self._render_counter += 1
        rgb_path = self.output_dir / f"agent_rgb_{self._render_counter:04d}.png"
        pca_path = self.output_dir / f"agent_pca_{self._render_counter:04d}.png"

        try:
            self._render_to_files(rgb_path, pca_path)
        except Exception as e:
            logger.exception("[SceneAgent] Rendering failed")
            # Create placeholder images so the agent loop can continue
            _create_placeholder_image(rgb_path, text="Render failed")
            _create_placeholder_image(pca_path, text="Render failed")

        return rgb_path, pca_path

    def _render_to_files(self, rgb_path: Path, pca_path: Path):
        """
        Render the scene from the first training viewpoint.

        Args:
            rgb_path: Output path for the RGB image.
            pca_path: Output path for the PCA semantic image.
        """
        # Construct a default camera batch from the system's training data
        batch = self._get_default_camera_batch()
        if batch is None:
            raise RuntimeError("No camera batch available for rendering")

        with torch.no_grad():
            output = self.system(batch)

        # Save RGB image
        rgb = output["comp_rgb"][0]  # [H, W, 3]
        rgb_np = (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(rgb_np).save(rgb_path)

        # Save PCA semantic image
        lang = output.get("comp_lang")
        if lang is not None:
            lang_np = lang[0].cpu().numpy()  # [H, W, d]
            pca_img = self._apply_pca_visualization(lang_np)
            Image.fromarray(pca_img).save(pca_path)
        else:
            _create_placeholder_image(pca_path, text="No semantic features")

    def _apply_pca_visualization(self, lang_features: np.ndarray) -> np.ndarray:
        """
        Convert rendered language features to RGB via PCA.

        For d=3, this reduces to a robust normalization (the PCA rotation
        aligns the semantic principal axes to RGB channels, maximizing
        color discrimination between different objects).

        Args:
            lang_features: Language feature map, shape [H, W, d].

        Returns:
            RGB image as uint8 numpy array, shape [H, W, 3].
        """
        from sklearn.decomposition import PCA

        h, w, d = lang_features.shape
        flat = lang_features.reshape(-1, d)

        # Fit PCA on the current frame
        pca = PCA(n_components=3)
        transformed = pca.fit_transform(flat)

        # Robust percentile normalization
        low = np.percentile(transformed, 1, axis=0)
        high = np.percentile(transformed, 99, axis=0)
        normalized = (transformed - low) / (high - low + 1e-9)
        normalized = np.clip(normalized, 0, 1)

        return (normalized.reshape(h, w, 3) * 255).astype(np.uint8)

    def _get_default_camera_batch(self) -> Optional[Dict[str, Any]]:
        """
        Get a default camera batch for rendering.

        Tries to use the first cached training view. Falls back to a
        synthetic front-facing camera if no training data is available.

        Returns:
            Camera batch dict, or None if unavailable.
        """
        # Try to use cached training batch
        if hasattr(self.system, "cached_batches") and self.system.cached_batches:
            batch = self.system.cached_batches[0]
            return {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

        # Try to construct from geometry bounds
        if hasattr(self.system.geometry, "get_xyz"):
            xyz = self.system.geometry.get_xyz.detach()
            center = xyz.mean(dim=0)
            extent = (xyz.max(dim=0).values - xyz.min(dim=0).values).max().item()

            # Construct a simple front-facing camera looking AT the center from -Z
            eye = center + torch.tensor(
                [0, 0, -extent * 1.5], device=self.device, dtype=torch.float32
            )
            
            # The c2w needs to rotate the camera to face +Z (towards the object) 
            # In standard CV convention, +Z is forward. So if we are at -Z looking at origin,
            # we need to point towards +Z. The identity matrix [0,0,1] is looking +Z.
            # But the camera convention might need Y up or down depending on the renderer.
            # Let's use a standard lookat or simple translation with identity rotation
            # if the renderer expects +Z forward and +Y down (common in 3DGS).
            c2w = torch.eye(4, device=self.device, dtype=torch.float32)
            # Flip Y and Z to look forward from negative Z position, with Y down
            c2w[1, 1] = -1.0
            c2w[2, 2] = -1.0
            
            c2w[:3, 3] = eye

            return {
                "c2w": c2w.unsqueeze(0),
                "fovy": torch.tensor([0.8], device=self.device),
                "width": 512,
                "height": 512,
            }

        return None

    # ------------------------------------------------------------------
    # LLM response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_function_calls(response):
        """
        Extract ALL function calls from a Gemini response.

        Gemini may return multiple function calls in parallel. This method
        extracts all of them rather than only the first one.

        Args:
            response: Gemini API response object.

        Returns:
            List of (function_name, args_dict) tuples, or None if no
            function calls are present.
        """
        calls = []
        try:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "function_call") and part.function_call.name:
                    fc = part.function_call
                    args = dict(fc.args) if fc.args else {}
                    calls.append((fc.name, args))
        except (AttributeError, IndexError, TypeError):
            pass
        return calls if calls else None

    @staticmethod
    def _extract_text(response) -> str:
        """
        Extract text content from a Gemini response.

        Args:
            response: Gemini API response object.

        Returns:
            Text string from the response.
        """
        try:
            return response.text
        except (AttributeError, ValueError):
            try:
                parts = response.candidates[0].content.parts
                texts = [p.text for p in parts if hasattr(p, "text") and p.text]
                return "\n".join(texts) if texts else "No response generated."
            except (AttributeError, IndexError):
                return "Failed to parse LLM response."

    def _send_multiple_tool_results(
        self,
        results: List[Tuple[str, Dict[str, Any]]],
    ):
        """
        Send multiple tool results back to the LLM in a single message.

        When Gemini returns parallel function calls, all results must be
        sent back together for the model to continue reasoning.

        Args:
            results: List of (function_name, result_dict) tuples.

        Returns:
            Gemini API response object for the next turn.
        """
        import google.generativeai as genai

        response_parts = []
        for func_name, result in results:
            sanitized = self._sanitize_result(result)
            response_parts.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=func_name,
                        response={"result": sanitized},
                    )
                )
            )

        return self.llm.chat_session.send_message(response_parts)

    @staticmethod
    def _sanitize_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove non-serializable entries (e.g. torch.Tensor) from result dict
        before sending to the LLM.

        Args:
            result: Raw result dict from editor operations.

        Returns:
            Cleaned dict with only JSON-serializable values.
        """
        sanitized = {}
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                # Skip tensor values (mask, etc.) — not needed by LLM
                continue
            sanitized[key] = value
        return sanitized


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def _create_placeholder_image(
    path: Path,
    text: str = "N/A",
    size: Tuple[int, int] = (512, 512),
):
    """
    Create a simple placeholder image with text.

    Args:
        path: Output file path.
        text: Text to display on the placeholder.
        size: Image size (width, height).
    """
    img = Image.new("RGB", size, color=(64, 64, 64))
    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.text((size[0] // 4, size[1] // 2), text, fill=(255, 255, 255))
    except ImportError:
        pass
    img.save(path)
