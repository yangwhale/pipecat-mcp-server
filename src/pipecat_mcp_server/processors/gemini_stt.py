"""Gemini multimodal STT for Pipecat (Vertex AI primary, API key fallback)."""

from __future__ import annotations

import os
from typing import AsyncGenerator, Optional

from google import genai
from google.genai import types as genai_types
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601


_DEFAULT_PROMPT = (
    "Transcribe this Chinese audio verbatim. "
    "Output only the transcription, no explanation, no punctuation other than "
    "what the speaker uses. "
    "If the audio contains no clear speech (silence, background noise only, "
    "or unintelligible sounds), output an empty string with nothing else."
)


class GeminiSTTService(SegmentedSTTService):
    def __init__(
        self,
        *,
        model: str = "gemini-3-flash-preview",
        language: str = "cmn-Hans-CN",
        prompt: str = _DEFAULT_PROMPT,
        project: Optional[str] = None,
        location: str = "global",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model = model
        self._language = language
        self._prompt = prompt
        self._client = self._build_client(project=project, location=location, api_key=api_key)

    @staticmethod
    def _build_client(
        *, project: Optional[str], location: str, api_key: Optional[str]
    ) -> genai.Client:
        try:
            client = genai.Client(
                vertexai=True,
                project=project or os.environ.get("GOOGLE_CLOUD_PROJECT"),
                location=location,
            )
            logger.info(
                f"GeminiSTT: using Vertex AI (project={client._api_client.project}, location={location})"
            )
            return client
        except Exception as e:
            logger.warning(f"GeminiSTT: Vertex AI init failed ({e}); falling back to Gemini API key")

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "GeminiSTT: Vertex AI unavailable and GEMINI_API_KEY not set. "
                "Run `gcloud auth application-default login` or set GEMINI_API_KEY."
            )
        logger.info("GeminiSTT: using Gemini API key fallback")
        return genai.Client(api_key=key)

    def can_generate_metrics(self) -> bool:
        return True

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        await self.start_processing_metrics()
        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=[
                    self._prompt,
                    genai_types.Part.from_bytes(data=audio, mime_type="audio/wav"),
                ],
            )
        except Exception as e:
            logger.error(f"GeminiSTT: generate_content failed: {e}")
            yield ErrorFrame(error=f"Gemini STT error: {e}")
            return
        finally:
            await self.stop_processing_metrics()

        text = (response.text or "").strip()
        if not text:
            return

        logger.debug(f"GeminiSTT transcription: [{text}]")
        yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), self._language)
