"""Gemini multimodal TTS for Pipecat (Vertex AI primary, API key fallback)."""

from __future__ import annotations

import os
from typing import AsyncGenerator, Optional

from google import genai
from google.genai import types as genai_types
from loguru import logger
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService


GEMINI_TTS_SAMPLE_RATE = 24000


class GeminiTTSService(TTSService):
    def __init__(
        self,
        *,
        model: str = "gemini-3.1-flash-tts-preview",
        voice: str = "Charon",
        project: Optional[str] = None,
        location: str = "global",
        api_key: Optional[str] = None,
        sample_rate: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate or GEMINI_TTS_SAMPLE_RATE, **kwargs)
        self._model = model
        self._voice = voice
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
                f"GeminiTTS: using Vertex AI (project={client._api_client.project}, location={location})"
            )
            return client
        except Exception as e:
            logger.warning(f"GeminiTTS: Vertex AI init failed ({e}); falling back to Gemini API key")

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "GeminiTTS: Vertex AI unavailable and GEMINI_API_KEY not set. "
                "Run `gcloud auth application-default login` or set GEMINI_API_KEY."
            )
        logger.info("GeminiTTS: using Gemini API key fallback")
        return genai.Client(api_key=key)

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"GeminiTTS: synthesizing [{text}]")

        config = genai_types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=genai_types.SpeechConfig(
                voice_config=genai_types.VoiceConfig(
                    prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(voice_name=self._voice)
                )
            ),
        )

        try:
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)
            yield TTSStartedFrame()

            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=text,
                config=config,
            )

            pcm = bytearray()
            for part in response.candidates[0].content.parts:
                inline = getattr(part, "inline_data", None)
                if inline and inline.data:
                    pcm.extend(inline.data)

            await self.stop_ttfb_metrics()

            if not pcm:
                yield ErrorFrame(error="GeminiTTS returned no audio data")
                return

            yield TTSAudioRawFrame(
                audio=bytes(pcm), sample_rate=self.sample_rate, num_channels=1
            )
        except Exception as e:
            logger.error(f"GeminiTTS: generate_content failed: {e}")
            yield ErrorFrame(error=f"Gemini TTS error: {e}")
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()
