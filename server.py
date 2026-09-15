"""Minimal language identification API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from lid import LanguageIdentifier, LanguageIdentifierError

# ---------------------------------------------------------------------------
# Change this to "cpu" if you do not have a working NVIDIA GPU / cuDNN setup.
# ---------------------------------------------------------------------------
DEVICE = "cuda"
# DEVICE = "cpu"

HOST = "0.0.0.0"
PORT = 8007
CANDIDATE_LANGUAGES = ["hi", "kn", "mr", "ta", "te"]
MARGIN_THRESHOLD = 0.050965
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _parse_languages(raw: str | None) -> list[str] | None:
    if raw is None or raw.strip() == "":
        return None
    parts = [p.strip() for p in raw.replace(";", ",").split(",")]
    return [p for p in parts if p]


def create_app(model: LanguageIdentifier | None = None) -> FastAPI:
    state: dict[str, Any] = {"model": model}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if state["model"] is None:
            try:
                state["model"] = LanguageIdentifier(
                    device=DEVICE,
                    candidate_languages=CANDIDATE_LANGUAGES,
                    margin_threshold=MARGIN_THRESHOLD,
                )
            except LanguageIdentifierError as exc:
                raise RuntimeError(f"Failed to load language ID model: {exc}") from exc
        yield

    app = FastAPI(title="Language Identification", version="1.0.0", lifespan=lifespan)

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

    @app.get("/health")
    def health() -> dict[str, Any]:
        try:
            return get_model().health()
        except HTTPException:
            return {
                "status": "error",
                "device": DEVICE,
                "model_loaded": False,
                "available_languages": [],
                "providers": [],
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
        return JSONResponse(
            {
                "language": result["language"],
                "scores": result["scores"],
                "margin": result["margin"],
                "top_candidates": result["top_candidates"],
            }
        )

    return app


app = create_app()


def main() -> None:
    import uvicorn

    print(f"Device={DEVICE}  UI=http://127.0.0.1:{PORT}/")
    uvicorn.run("server:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
