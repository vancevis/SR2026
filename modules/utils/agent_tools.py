"""
Agent Tool Definitions Module
Defines the tool schemas for Gemini Function Calling in the scene editing agent.
Each tool corresponds to a specific scene editing or querying capability.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def get_tool_declarations() -> List[Any]:
    """
    Build and return all tool declarations for Gemini Function Calling.

    Returns:
        List of genai.protos.Tool wrapping all function declarations.
    """
    if not GEMINI_AVAILABLE:
        logger.warning("google.generativeai not available, returning empty tools")
        return []

    declarations = [
        _decl_select_and_highlight(),
        _decl_translate_object(),
        _decl_rotate_object(),
        _decl_scale_object(),
        _decl_delete_object(),
        _decl_describe_scene(),
        _decl_undo_last_edit(),
        _decl_render_current_scene(),
        _decl_evaluate_semantic_consistency(),
        _decl_adaptive_finetune(),
    ]

    return [genai.protos.Tool(function_declarations=declarations)]


# ---------------------------------------------------------------------------
# Individual tool declarations
# ---------------------------------------------------------------------------

def _decl_select_and_highlight() -> Any:
    """Tool declaration: select and highlight an object by language description."""
    return genai.protos.FunctionDeclaration(
        name="select_and_highlight",
        description=(
            "Select an object in the 3D scene by its natural-language description. "
            "Returns the number of selected Gaussians, the object center coordinates, "
            "and the axis-aligned bounding box. Use this tool FIRST to verify that "
            "the target object can be found before performing any editing operation."
        ),
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "object_description": genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    description="Natural-language description of the target object, e.g. 'red chair' or 'the lamp on the table'.",
                ),
                "threshold": genai.protos.Schema(
                    type=genai.protos.Type.NUMBER,
                    description="Similarity threshold for selection (0.0-1.0). Lower values select more Gaussians. Default is 0.5.",
                ),
            },
            required=["object_description"],
        ),
    )


def _decl_translate_object() -> Any:
    """Tool declaration: translate (move) a selected object."""
    return genai.protos.FunctionDeclaration(
        name="translate_object",
        description=(
            "Move an object in the 3D scene along the x, y, z axes. "
            "Positive x is right, positive y is up, positive z is forward. "
            "Typical small movements are 0.1-0.5 units."
        ),
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "object_description": genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    description="Natural-language description of the object to move.",
                ),
                "dx": genai.protos.Schema(
                    type=genai.protos.Type.NUMBER,
                    description="Translation along x-axis (right is positive).",
                ),
                "dy": genai.protos.Schema(
                    type=genai.protos.Type.NUMBER,
                    description="Translation along y-axis (up is positive).",
                ),
                "dz": genai.protos.Schema(
                    type=genai.protos.Type.NUMBER,
                    description="Translation along z-axis (forward is positive).",
                ),
            },
            required=["object_description", "dx", "dy", "dz"],
        ),
    )


def _decl_rotate_object() -> Any:
    """Tool declaration: rotate a selected object."""
    return genai.protos.FunctionDeclaration(
        name="rotate_object",
        description=(
            "Rotate an object in the 3D scene. Angles are specified in degrees. "
            "The rotation is applied around the object's center of mass."
        ),
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "object_description": genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    description="Natural-language description of the object to rotate.",
                ),
                "roll": genai.protos.Schema(
                    type=genai.protos.Type.NUMBER,
                    description="Rotation around x-axis in degrees.",
                ),
                "pitch": genai.protos.Schema(
                    type=genai.protos.Type.NUMBER,
                    description="Rotation around y-axis in degrees.",
                ),
                "yaw": genai.protos.Schema(
                    type=genai.protos.Type.NUMBER,
                    description="Rotation around z-axis in degrees.",
                ),
            },
            required=["object_description", "roll", "pitch", "yaw"],
        ),
    )


def _decl_scale_object() -> Any:
    """Tool declaration: scale a selected object."""
    return genai.protos.FunctionDeclaration(
        name="scale_object",
        description=(
            "Scale an object in the 3D scene. A scale_factor of 1.0 means no change, "
            "2.0 means double size, 0.5 means half size."
        ),
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "object_description": genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    description="Natural-language description of the object to scale.",
                ),
                "scale_factor": genai.protos.Schema(
                    type=genai.protos.Type.NUMBER,
                    description="Scaling factor. Must be positive (e.g. 1.5 = 150%).",
                ),
            },
            required=["object_description", "scale_factor"],
        ),
    )


def _decl_delete_object() -> Any:
    """Tool declaration: delete an object and optionally inpaint the background."""
    return genai.protos.FunctionDeclaration(
        name="delete_object",
        description=(
            "Remove an object from the 3D scene. Optionally inpaints the background "
            "region using a diffusion model to fill in the hole left by the deleted object."
        ),
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "object_description": genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    description="Natural-language description of the object to delete.",
                ),
                "inpaint_background": genai.protos.Schema(
                    type=genai.protos.Type.BOOLEAN,
                    description="Whether to inpaint the background after deletion. Default is True.",
                ),
            },
            required=["object_description"],
        ),
    )


def _decl_describe_scene() -> Any:
    """Tool declaration: answer a question about the scene content."""
    return genai.protos.FunctionDeclaration(
        name="describe_scene",
        description=(
            "Answer a question about the 3D scene content by analyzing the RGB image "
            "and semantic feature map. Use this for scene understanding queries like "
            "'how many chairs are there?' or 'what is next to the sofa?'."
        ),
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "question": genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    description="The question to answer about the scene.",
                ),
            },
            required=["question"],
        ),
    )


def _decl_undo_last_edit() -> Any:
    """Tool declaration: undo the most recent editing operation."""
    return genai.protos.FunctionDeclaration(
        name="undo_last_edit",
        description=(
            "Undo the most recent editing operation and restore the scene to its "
            "previous state. Only works if there is an edit to undo."
        ),
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={},
        ),
    )


def _decl_render_current_scene() -> Any:
    """Tool declaration: render the current scene from a specified viewpoint."""
    return genai.protos.FunctionDeclaration(
        name="render_current_scene",
        description=(
            "Render the current state of the 3D scene and return the image path. "
            "Use this to visually verify the result after an editing operation."
        ),
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "description": genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    description="Optional description for logging purposes.",
                ),
            },
        ),
    )


def _decl_evaluate_semantic_consistency() -> Any:
    """Tool declaration: evaluate semantic consistency after an edit."""
    return genai.protos.FunctionDeclaration(
        name="evaluate_semantic_consistency",
        description=(
            "Evaluate the semantic consistency of the 3D scene after an editing "
            "operation. Returns quantitative metrics including:\n"
            "- SCS (Semantic Consistency Score): How well the semantic field "
            "matches training targets (0-1, higher is better).\n"
            "- URP (Unedited Region Preservation): Stability of non-edited "
            "regions (0-1, higher is better).\n"
            "- FDS (Feature Distribution Shift): Abnormal feature drift "
            "(>=0, lower is better; >0.5 indicates artifacts).\n"
            "- recommended_finetune_steps: Suggested number of semantic "
            "fine-tuning steps to restore consistency.\n\n"
            "IMPORTANT: Call this tool AFTER every editing operation to assess "
            "whether the edit maintained semantic coherence. If SCS < 0.7 or "
            "FDS > 0.5, consider calling adaptive_finetune or undo_last_edit."
        ),
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={},
        ),
    )


def _decl_adaptive_finetune() -> Any:
    """Tool declaration: adaptively fine-tune semantic features post-edit."""
    return genai.protos.FunctionDeclaration(
        name="adaptive_finetune",
        description=(
            "Run adaptive semantic fine-tuning on the edited region of the 3D "
            "scene. This restores semantic consistency after geometric edits "
            "(translate, rotate, scale). The number of fine-tuning steps is "
            "determined automatically based on the semantic evaluation metrics, "
            "but can be overridden.\n\n"
            "Call this tool when evaluate_semantic_consistency reports low SCS "
            "or high FDS. It is NOT needed after delete operations."
        ),
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "override_steps": genai.protos.Schema(
                    type=genai.protos.Type.INTEGER,
                    description=(
                        "Override the automatically computed number of fine-tuning "
                        "steps. If not specified, uses the recommended steps from "
                        "the last semantic evaluation. Typical range: 10-200."
                    ),
                ),
                "learning_rate": genai.protos.Schema(
                    type=genai.protos.Type.NUMBER,
                    description=(
                        "Learning rate for fine-tuning. Default is 0.005. "
                        "Increase to 0.01 for aggressive correction, decrease "
                        "to 0.001 for gentle adjustment."
                    ),
                ),
            },
        ),
    )
