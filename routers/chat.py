import logging
from typing import Annotated
import httpx

from fastapi import Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRouter

from dependencies import LibraryDep, TemplatesDep

router = APIRouter(prefix="/chat")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize conversation history
# TODO Make non-global
conversation_history: dict[str, list[dict[str, str]]] = {}

# Add a constant for the model name
# TODO move to optional env var
OLLAMA_MODEL = "llama3.1:latest"  # Changed from "mistral" to "llama2:13b"
OLLAMA_API_BASE = "http://localhost:11434"  # Default Ollama API endpoint


@router.get("/{course_name}/{topic_name}/{activity_name}", response_model=None)
async def chat(
    request: Request,
    library: LibraryDep,
    templates: TemplatesDep,
    course_name: str,
    topic_name: str,
    activity_name: str,
) -> HTMLResponse:
    try:
        logger.info("Generating prompt for chat")
        initial_prompt = library.generate_prompt(course_name, topic_name, activity_name)
        logger.debug(f"Generated initial prompt: {initial_prompt}")

        # Get the topic text
        topic = library.get_topic(course_name, topic_name)
        if not topic:
            logger.error(f"Topic not found: {course_name}/{topic_name}")
            raise HTTPException(status_code=404, detail="Topic not found")

        if initial_prompt:
            # Initialize conversation with system prompt
            conversation_key = f"{course_name}_{topic_name}_{activity_name}"

            conversation_history.update(
                {conversation_key: [{"role": "system", "content": initial_prompt}]}
            )

            # Get first response from AI
            messages = conversation_history.get(conversation_key)
            if not messages:
                logger.error("No messages found in conversation history")
                raise HTTPException(status_code=500, detail="Conversation initialization failed")

            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    logger.debug(f"Sending request to Ollama API with messages: {messages}")
                    response = await client.post(
                        f"{OLLAMA_API_BASE}/api/chat",
                        json={
                            "model": OLLAMA_MODEL,
                            "messages": messages,
                        }
                    )
                    response.raise_for_status()
                    chat_completion = response.json()
                    logger.debug(f"Received response from Ollama: {chat_completion}")

            except httpx.HTTPError as e:
                logger.error(f"HTTP error occurred: {e}")
                raise HTTPException(status_code=502, detail=f"Ollama API error: {str(e)}")
            except Exception as e:
                logger.error(f"Error calling Ollama API: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get AI response: {str(e)}")

            try:
                initial_response = chat_completion["message"]["content"]
                if initial_response is None:
                    raise ValueError("No response content from Ollama")

                initial_response = initial_response.strip()

                # Add AI's response to conversation history
                conversation_history[conversation_key].append(
                    {"role": "assistant", "content": initial_response}
                )

                return templates.TemplateResponse(
                    "chat.html",
                    {
                        "request": request,
                        "initial_prompt": initial_response,
                        "course_name": course_name,
                        "topic_name": topic_name,
                        "activity_name": activity_name,
                    },
                )
            except (KeyError, ValueError) as e:
                logger.error(f"Error processing Ollama response: {e}")
                raise HTTPException(status_code=500, detail=f"Invalid response from Ollama: {str(e)}")

        logger.error("No initial prompt generated")
        raise HTTPException(status_code=404, detail="Invalid parameters")

    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=None)
async def chat_post(
    request: Request,
    library: LibraryDep,
    templates: TemplatesDep,
    message: Annotated[str, Form()],
    course_name: Annotated[str, Form()],
    topic_name: Annotated[str, Form()],
    activity_name: Annotated[str, Form()],
) -> HTMLResponse:
    global conversation_history

    try:
        logger.info(f"Received message: {message}")

        conversation_key = f"{course_name}_{topic_name}_{activity_name}"

        # Initialize conversation history if it doesn't exist
        if conversation_key not in conversation_history:
            system_message = {
                "role": "system",
                "content": library.generate_prompt(course_name, topic_name, activity_name),
            }
            conversation_history[conversation_key] = [system_message]

        # Add user message to history
        conversation_history[conversation_key].append(
            {"role": "user", "content": message}
        )

        # Get chat completion from Ollama
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OLLAMA_API_BASE}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": conversation_history[conversation_key],
                    "options": {
                        "temperature": 0.7,
                    }
                }
            )
            response.raise_for_status()
            chat_completion = response.json()

        # Extract the response
        ai_response = chat_completion["message"]["content"]

        # Add AI response to history
        conversation_history[conversation_key].append(
            {"role": "assistant", "content": ai_response}
        )

        logger.debug(f"Rendering template with user_message: {message}")
        logger.debug(f"Rendering template with ai_response: {ai_response}")

        return templates.TemplateResponse(
            "chat_messages.html",
            {
                "request": request,
                "user_message": message,
                "ai_response": ai_response,
            },
        )

    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
