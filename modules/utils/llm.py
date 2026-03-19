"""
LLM Interaction Module
Integration with Google Gemini for 3D scene understanding and interaction
"""

import os
import time
import warnings
from typing import List, Optional, Tuple
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning, module="google.*")
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class SceneLLM:
    """
    LLM Agent for 3D Scene Interaction using Gemini
    """
    
    def __init__(self, api_key: str = None, model_name: str = "gemini-3.0-pro"):
        if not GEMINI_AVAILABLE:
            self.model = None
            return

        # Try to get API key from env if not provided
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
        
        if not api_key:
            self.model = None
            return
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Failed to initialize Gemini: %s", e)
            self.model = None
            return
        self.chat_history = []

    def parse_command(self, query: str) -> str:
        """
        Parse user command to extract object description for CLIP
        Example: "Where is the red chair?" -> "red chair"
        """
        if not self.model:
            return query
            
        prompt = f"""
        Extract the main object description from this user query for visual search.
        Return ONLY the object description.
        
        Query: {query}
        Object:
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return query

    def chat_with_scene(self, query: str, image_paths: List[str]) -> str:
        """
        Chat with scene using RGB and semantic PCA images
        Enhanced with system prompt for better object recognition
        
        Args:
            query: User question about the scene
            image_paths: List of image paths [RGB, semantic PCA, ...]
        
        Returns:
            LLM response text
        """
        if not self.model:
            return "LLM not available."
        
        # Load images
        images = []
        for p in image_paths:
            try:
                img = Image.open(str(p))
                images.append(img)
            except Exception as e:
                print(f"Warning: Failed to load {p}")
                continue
        
        if not images:
            return "No scene images available."
        
        # Enhanced system prompt for semantic understanding
        system_prompt = """You are an intelligent 3D Scene Assistant. You are provided with images:
1. RGB image showing the real appearance
2. Semantic Feature Map where different objects have distinct colors (e.g., chair is purely green, table is red)

Use the Semantic Feature Map to clearly distinguish object boundaries and count objects.
Use the RGB image to identify what they are.

Answer concisely and accurately."""
        
        # Construct prompt
        prompt_content = [system_prompt, f"User Question: {query}"] + images
        
        try:
            response = self.model.generate_content(prompt_content)
            return response.text
        except Exception as e:
            return f"Error: {e}"


# Agent system prompt — instructs the LLM on its role and capabilities
AGENT_SYSTEM_PROMPT = (
    "You are an intelligent 3D Scene Editing Agent with semantic-aware "
    "closed-loop feedback. You can see the scene through RGB and semantic "
    "feature images provided to you.\n\n"
    "You have access to tools for selecting, moving, rotating, scaling, "
    "deleting objects, AND for evaluating and repairing semantic consistency "
    "in a 3D scene.\n\n"
    "When the user gives you an editing instruction, follow this workflow:\n"
    "1. First understand the current scene by looking at the images.\n"
    "2. Identify the target object(s) using select_and_highlight.\n"
    "3. Plan the editing steps needed.\n"
    "4. Execute the edit using the appropriate tool.\n"
    "5. ALWAYS call evaluate_semantic_consistency after any edit.\n"
    "6. Interpret the metrics:\n"
    "   - SCS (Semantic Consistency Score): 0-1, higher=better. "
    "If < 0.7, semantic repair is needed.\n"
    "   - URP (Unedited Region Preservation): 0-1, higher=better. "
    "If < 0.8, the edit disturbed nearby regions.\n"
    "   - FDS (Feature Distribution Shift): >=0, lower=better. "
    "If > 0.5, artifacts are likely.\n"
    "   - MVC (Multi-View Consistency): 0-1, higher=better. "
    "If < 0.8, some views are much worse than others; "
    "check worst_view_SCS.\n"
    "   - recommended_finetune_steps: auto-computed repair steps.\n"
    "7. If SCS < 0.7 or FDS > 0.5, call adaptive_finetune to repair "
    "semantic consistency. You may adjust learning_rate or override_steps "
    "based on the severity.\n"
    "8. If SCS < 0.4 after finetune, consider undo_last_edit and retrying "
    "with different parameters.\n"
    "9. Render the final result and show it to the user.\n\n"
    "When the user asks a question about the scene, use describe_scene.\n"
    "Always explain what you are doing, show metrics, and show results.\n"
    "Respond in the same language as the user."
)


class SceneAgentLLM(SceneLLM):
    """
    Extended LLM with Gemini Function Calling support.

    Builds on SceneLLM to provide:
    - Function Calling tool declarations for scene editing
    - Multi-turn chat session with persistent context
    - Tool result feedback mechanism

    Attributes:
        tools: Gemini tool declarations for Function Calling.
        chat_session: Active multi-turn chat session.
    """

    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gemini-3.0-pro",
    ):
        """
        Initialize SceneAgentLLM with Function Calling capabilities.

        Args:
            api_key: Google API key. If None, reads from GOOGLE_API_KEY env.
            model_name: Gemini model name to use.
        """
        # Initialize base class for API key setup
        super().__init__(api_key, model_name)

        if not GEMINI_AVAILABLE or self.model is None:
            self.tools = None
            self.chat_session = None
            return

        # Load tool declarations
        from .agent_tools import get_tool_declarations
        self.tools = get_tool_declarations()

        # Create model with tool support and system instruction
        self.model = genai.GenerativeModel(
            model_name,
            tools=self.tools,
            system_instruction=AGENT_SYSTEM_PROMPT,
        )

        # Start a multi-turn chat session
        self.chat_session = self.model.start_chat(
            enable_automatic_function_calling=False,
        )

    def send_message(
        self,
        text: str = None,
        image_paths: list = None,
    ):
        """
        Send a message (text and/or images) to the LLM chat session.

        Args:
            text: Text message content. Can be None in follow-up turns.
            image_paths: List of image file paths to include as visual context.

        Returns:
            Gemini API response object (may contain text or function_call).
        """
        if self.chat_session is None:
            raise RuntimeError("SceneAgentLLM not initialized (missing API key)")

        content = []

        # Attach images if provided
        if image_paths:
            for img_path in image_paths:
                try:
                    img = Image.open(str(img_path))
                    content.append(img)
                except Exception:
                    logging.getLogger(__name__).warning(
                        "Failed to load image: %s", img_path
                    )

        # Attach text
        if text:
            content.append(text)

        if not content:
            raise ValueError("At least one of text or image_paths must be provided")

        return self.chat_session.send_message(content)

    def send_tool_result(self, function_name: str, result: dict):
        """
        Send a tool execution result back to the LLM.

        After the LLM requests a function call, the caller executes the
        function and sends the result back via this method. The LLM then
        continues reasoning with the new information.

        Args:
            function_name: Name of the function that was called.
            result: Dict containing the function's return value.

        Returns:
            Gemini API response object for the next turn.
        """
        if self.chat_session is None:
            raise RuntimeError("SceneAgentLLM not initialized (missing API key)")

        response_part = genai.protos.Part(
            function_response=genai.protos.FunctionResponse(
                name=function_name,
                response={"result": result},
            )
        )
        return self.chat_session.send_message(response_part)

    def reset_session(self):
        """
        Reset the chat session, clearing all conversation history.

        This is useful when starting a new editing task from scratch.
        """
        if self.model is not None:
            self.chat_session = self.model.start_chat(
                enable_automatic_function_calling=False,
            )
