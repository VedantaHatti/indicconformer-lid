"""FastAPI language identification service (no transcription)."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from lid import LanguageIdentifier, LanguageIdentifierError

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _parse_languages(raw: str | None) -> list[str] | None:
    if raw is None or raw.strip() == "":
        return None
    parts = [p.strip() for p in raw.replace(";", ",").split(",")]
    return [p for p in parts if p]


def _config_from_env() -> dict[str, Any]:
    languages = _parse_languages(os.getenv("LID_CANDIDATE_LANGUAGES", "hi,kn,mr,ta,te"))
    return {
        "model_dir": Path(os.getenv("MODEL_DIR", "model")).expanduser(),
        "device": os.getenv("LID_DEVICE", "cuda").strip().lower(),
        "candidate_languages": languages,
        "margin_threshold": float(os.getenv("LID_MARGIN_THRESHOLD", "0.050965")),
    }


def create_app(model: LanguageIdentifier | None = None) -> FastAPI:
    state: dict[str, Any] = {"model": model}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if state["model"] is None:
            cfg = _config_from_env()
            try:
                state["model"] = LanguageIdentifier(**cfg)
            except LanguageIdentifierError as exc:
                raise RuntimeError(f"Failed to load language ID model: {exc}") from exc
        yield

    app = FastAPI(
        title="Language Identification",
        description=(
            "Spoken language identification using a shared Conformer encoder, "
            "shared CTC output, and language vocabulary masks. Does not transcribe."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    def get_model() -> LanguageIdentifier:
        loaded = state.get("model")
        if loaded is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        return loaded

    @app.get("/")
    def ui() -> FileResponse:
        index = STATIC_DIR / "index.html"
        if not index.is_file():
            raise HTTPException(status_code=404, detail="Frontend not found")
        return FileResponse(index)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/health")
    def health() -> dict[str, Any]:
        try:
            return get_model().health()
        except HTTPException:
            cfg = _config_from_env()
            return {
                "status": "error",
                "model_loaded": False,
                "device": cfg["device"],
                "providers": [],
                "languages_available": [],
            }

    @app.post("/identify")
    async def identify(
        audio: UploadFile = File(..., description="Spoken audio file"),
        candidate_languages: str | None = Form(default=None),
    ) -> JSONResponse:
        payload = await audio.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Empty audio upload")
        languages = _parse_languages(candidate_languages)
        try:
            result = get_model().identify(payload, candidate_languages=languages)
        except (LanguageIdentifierError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        result.pop("transcript", None)
        result.pop("text", None)
        return JSONResponse(result)

    return app


app = create_app()


def main() -> None:
    import uvicorn

    host = os.getenv("LID_HOST", "0.0.0.0")
    port = int(os.getenv("LID_PORT", "8007"))
    uvicorn.run("server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
