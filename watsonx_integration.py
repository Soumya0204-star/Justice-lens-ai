"""
watsonx_integration.py
========================
Core IBM watsonx.ai integration layer for JusticeLens AI.

Provides a thin, dependency-light REST client for IBM watsonx.ai's
Foundation Models text-generation API (IBM Granite models), built directly
on ``requests`` rather than the full ``ibm-watsonx-ai`` SDK so that:

    * The credential/auth flow (IBM Cloud IAM API-key -> bearer token
      exchange -> ``/ml/v1/text/generation`` call) is fully transparent
      and auditable.
    * The module has no hard dependency beyond what the rest of the
      backend already requires (``requests``), while remaining a drop-in
      target for the official SDK later if desired.

All credentials (``WATSONX_API_KEY``, ``WATSONX_PROJECT_ID``, etc.) are
read exclusively from environment variables via ``config.WATSONX_CONFIG``,
which in turn loads them from a local ``.env`` file
(see ``.env.example``). **Nothing is ever hard-coded.**

Reliability contract (per the system architecture's NFRs): every narration
feature built on top of this client (executive summaries, district
comparisons, policy recommendations, the AI report generator, and the
Q&A engine) must keep working -- with a clearly-labeled, deterministic,
template-based fallback -- even when watsonx.ai is unreachable,
misconfigured, or rate-limited. This module therefore exposes
``GraniteClient.generate()`` as a call that raises a single,
well-typed ``WatsonxIntegrationError`` on any failure, which every
higher-level generator in this package catches and handles explicitly
(never lets a network hiccup crash the dashboard).

IBM Bob note
------------
IBM Bob is a development-time AI SDLC assistant (code generation, test
scaffolding, documentation, review) rather than a runtime inference API,
so it has no client class here analogous to ``GraniteClient``. Its
appropriate integration point for this project's ML workflows is
documented in ``docs/ibm_bob_workflow.md``: using it during development to
scaffold/review the model_training.py / model_evaluation.py /
shap_explainability.py modules, generate unit tests, and draft model-card
documentation for the best-selected model. It is not called at runtime by
this module or any other in this package.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

import requests

from justicelens import config
from justicelens.logger import get_logger
from justicelens.utils import WatsonxIntegrationError

logger = get_logger(__name__)


@dataclass
class NarrativeResult:
    """Standardized result shape returned by every generator built on top
    of ``GraniteClient`` (executive summary, district comparison, policy
    recommendation, AI report sections, and Q&A answers).

    Having one shared result type across all five generator modules keeps
    their public contracts identical regardless of whether a given call
    was actually served by Granite or fell back to a deterministic
    template -- callers (e.g. the Streamlit dashboard) can render
    ``narrative_text`` and ``is_ai_generated`` uniformly everywhere.

    Attributes:
        narrative_text: The final human-readable text to display.
        is_ai_generated: ``True`` if this text came from a live watsonx.ai
            /Granite call; ``False`` if it was produced by the
            deterministic template fallback (e.g. watsonx.ai was
            unreachable or not configured).
        model_id: The Granite model used, or ``"template_fallback"`` when
            ``is_ai_generated`` is ``False``.
        generated_at: UTC ISO-8601 timestamp of generation, for
            governance/audit logging.
        latency_seconds: Time taken to produce the narrative (0.0 for
            template fallback, since it's a local computation).
        source_data: The exact structured data the narrative was grounded
            in, preserved for audit purposes (so a reviewer can verify
            every number in ``narrative_text`` traces back to something
            computed deterministically upstream).
    """

    narrative_text: str
    is_ai_generated: bool
    model_id: str
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    latency_seconds: float = 0.0
    source_data: Dict[str, object] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Result of a single text-generation call to watsonx.ai.

    Attributes:
        text: The generated text, stripped of leading/trailing whitespace.
        model_id: The Granite (or other foundation) model that produced
            the text.
        input_token_count: Number of tokens in the prompt, if reported by
            the API (``None`` if not returned).
        generated_token_count: Number of tokens generated, if reported.
        stop_reason: Why generation stopped (e.g. "eos_token",
            "max_tokens"), if reported.
        latency_seconds: Wall-clock time the API call took.
    """

    text: str
    model_id: str
    input_token_count: Optional[int]
    generated_token_count: Optional[int]
    stop_reason: Optional[str]
    latency_seconds: float


class _IAMTokenManager:
    """Manages IBM Cloud IAM bearer tokens: exchanges the long-lived API
    key for a short-lived access token and transparently refreshes it
    before expiry.

    Not intended for direct external use -- ``GraniteClient`` owns one
    instance internally.
    """

    #: Refresh the token this many seconds before its reported expiry, to
    #: avoid racing a request against an about-to-expire token.
    _EXPIRY_SAFETY_MARGIN_SECONDS = 60

    def __init__(self, api_key: str, iam_url: str, timeout_seconds: float) -> None:
        """Initialize the token manager.

        Args:
            api_key: IBM Cloud API key (Service ID / user API key with
                access to the watsonx.ai project).
            iam_url: IBM Cloud IAM token endpoint.
            timeout_seconds: HTTP request timeout for the token exchange.

        Raises:
            WatsonxIntegrationError: If ``api_key`` is empty.
        """
        if not api_key:
            raise WatsonxIntegrationError(
                "Cannot create an IAM token manager without an API key. "
                "Set WATSONX_API_KEY in your .env file."
            )
        self._api_key = api_key
        self._iam_url = iam_url
        self._timeout_seconds = timeout_seconds
        self._cached_token: Optional[str] = None
        self._token_expires_at: float = 0.0

    def get_token(self) -> str:
        """Return a valid IAM bearer token, reusing the cached one when it
        has not yet neared expiry, otherwise fetching a fresh one.

        Returns:
            The bearer token string (without the "Bearer " prefix).

        Raises:
            WatsonxIntegrationError: If the IAM token exchange fails.
        """
        now = time.monotonic()
        if self._cached_token and now < self._token_expires_at:
            return self._cached_token

        try:
            response = requests.post(
                self._iam_url,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={
                    "apikey": self._api_key,
                    "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                },
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise WatsonxIntegrationError(
                f"IBM Cloud IAM token exchange failed (network error): {exc}"
            ) from exc
        except ValueError as exc:
            raise WatsonxIntegrationError(
                f"IBM Cloud IAM token exchange returned non-JSON response: {exc}"
            ) from exc

        token = payload.get("access_token")
        expires_in = payload.get("expires_in", 3600)
        if not token:
            raise WatsonxIntegrationError(
                f"IBM Cloud IAM token exchange response missing "
                f"'access_token'. Response keys: {list(payload.keys())}"
            )

        self._cached_token = token
        self._token_expires_at = now + int(expires_in) - self._EXPIRY_SAFETY_MARGIN_SECONDS
        logger.debug("Obtained new IAM token, valid for ~%ss", expires_in)
        return token


class GraniteClient:
    """REST client for IBM watsonx.ai Foundation Model text generation
    using IBM Granite models.

    Typical usage::

        client = GraniteClient()
        if client.is_available():
            result = client.generate("Summarize district X's disparity profile...")
            print(result.text)

    All higher-level generators in this package (executive summary,
    district comparison, policy recommendation, AI report, Q&A) hold one
    ``GraniteClient`` instance and are responsible for catching
    ``WatsonxIntegrationError`` and falling back to a deterministic
    template -- this client itself does not silently swallow failures, so
    callers always know definitively whether generation succeeded.
    """

    def __init__(self, watsonx_config: Optional[config.WatsonxConfig] = None) -> None:
        """Initialize the client.

        Args:
            watsonx_config: Configuration to use. Defaults to the
                module-level ``config.WATSONX_CONFIG`` singleton (which is
                populated from environment variables / ``.env``).
        """
        self._config = watsonx_config or config.WATSONX_CONFIG
        self._token_manager: Optional[_IAMTokenManager] = None
        if self._config.is_configured():
            self._token_manager = _IAMTokenManager(
                api_key=self._config.api_key,
                iam_url=self._config.iam_url,
                timeout_seconds=self._config.timeout_seconds,
            )

    def is_available(self) -> bool:
        """Check whether this client has the minimum credentials required
        to attempt a live watsonx.ai call.

        Returns:
            ``True`` if ``WATSONX_API_KEY`` and ``WATSONX_PROJECT_ID`` are
            both set; ``False`` otherwise (callers should use this to
            decide whether to even attempt :meth:`generate` before
            falling back to a template).
        """
        return self._config.is_configured()

    def _build_generation_payload(
        self,
        prompt: str,
        model_id: Optional[str],
        max_new_tokens: Optional[int],
        temperature: Optional[float],
        stop_sequences: Optional[list],
    ) -> Dict[str, object]:
        """Assemble the JSON body for a ``/ml/v1/text/generation`` call.

        Args:
            prompt: The full prompt text.
            max_new_tokens: Override for maximum generated tokens.
            temperature: Override for sampling temperature.
            stop_sequences: Optional list of strings that stop generation.

        Returns:
            The request body dict.
        """
        return {
            "model_id": model_id or self._config.model_id,
            "input": prompt,
            "parameters": {
                "decoding_method": "greedy" if (temperature or self._config.temperature) <= 0.0
                else "sample",
                "max_new_tokens": max_new_tokens or self._config.max_new_tokens,
                "min_new_tokens": 1,
                "temperature": temperature if temperature is not None else self._config.temperature,
                "repetition_penalty": 1.05,
                **({"stop_sequences": stop_sequences} if stop_sequences else {}),
            },
            **({"space_id": self._config.space_id} if getattr(self._config, "space_id", "") else {}),
            **({"project_id": self._config.project_id} if self._config.project_id else {}),
        }

    def _call_generation_endpoint(self, payload: Dict[str, object], token: str) -> Dict:
        """Perform the actual HTTP POST to the watsonx.ai text-generation
        endpoint, retrying transient failures with exponential backoff up
        to ``self._config.max_retries`` times.

        Args:
            payload: Request body built by
                :meth:`_build_generation_payload`.
            token: Valid IAM bearer token.

        Returns:
            The parsed JSON response body.

        Raises:
            WatsonxIntegrationError: On HTTP error, timeout, or malformed
                response, after all retries are exhausted.
        """
        url = f"{self._config.url}/ml/v1/text/generation"
        last_error: Optional[BaseException] = None

        for attempt in range(self._config.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    params={"version": self._config.api_version},
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    json=payload,
                    timeout=self._config.timeout_seconds,
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                error_text = ""
                response_obj = getattr(exc, "response", None)
                if response_obj is not None:
                    try:
                        error_text = response_obj.text[:1000]
                    except Exception:  # noqa: BLE001
                        error_text = ""
                last_error = WatsonxIntegrationError(
                    f"watsonx.ai request failed: {exc}{f' | response={error_text}' if error_text else ''}"
                )
            except ValueError as exc:
                last_error = WatsonxIntegrationError(
                    f"watsonx.ai returned invalid JSON: {exc}"
                )

            if attempt < self._config.max_retries:
                delay = 1.5 * (2**attempt)
                logger.warning(
                    "watsonx.ai generation request failed (attempt %d/%d): "
                    "%s. Retrying in %.1fs.",
                    attempt + 1,
                    self._config.max_retries + 1,
                    last_error,
                    delay,
                )
                time.sleep(delay)

        raise WatsonxIntegrationError(
            f"watsonx.ai text-generation request failed after "
            f"{self._config.max_retries + 1} attempt(s): {last_error}"
        ) from last_error

    def _candidate_model_ids(self) -> list[str]:
        candidates = list(dict.fromkeys([self._config.model_id, *config.WATSONX_MODEL_CANDIDATES]))
        return [candidate for candidate in candidates if candidate]

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[list] = None,
    ) -> GenerationResult:
        """Generate text from a prompt using the configured Granite model.

        Args:
            prompt: The full prompt string. Callers (the generator
                classes in this package) are responsible for building
                this via ``prompt_templates.py`` so that it is always
                grounded in precomputed structured data rather than
                asking the model to invent numbers.
            max_new_tokens: Override for maximum tokens generated.
            temperature: Override for sampling temperature (0.0 = greedy
                /deterministic decoding).
            stop_sequences: Optional list of stop strings.

        Returns:
            A populated :class:`GenerationResult`.

        Raises:
            WatsonxIntegrationError: If credentials are missing, the IAM
                token exchange fails, or the generation call fails after
                retries. Callers should catch this and fall back to a
                deterministic template rather than letting it propagate
                to the UI layer.
        """
        if not self.is_available():
            raise WatsonxIntegrationError(
                "watsonx.ai is not configured: set WATSONX_API_KEY and a "
                "WATSONX_PROJECT_ID or WATSONX_SPACE_ID in your .env file."
            )

        assert self._token_manager is not None  # guaranteed by is_available() check
        token = self._token_manager.get_token()
        start = time.monotonic()
        response_json = None
        last_error: Optional[BaseException] = None

        for model_id in self._candidate_model_ids():
            payload = self._build_generation_payload(
                prompt, model_id, max_new_tokens, temperature, stop_sequences
            )
            try:
                response_json = self._call_generation_endpoint(payload, token)
                break
            except WatsonxIntegrationError as exc:
                last_error = exc
                if "403" not in str(exc) and "Forbidden" not in str(exc):
                    raise
                logger.warning(
                    "watsonx.ai model '%s' was forbidden; trying the next candidate.",
                    model_id,
                )

        if response_json is None:
            raise WatsonxIntegrationError(
                f"watsonx.ai text-generation failed for all configured model candidates: {last_error}"
            )

        latency = time.monotonic() - start

        results = response_json.get("results", [])
        if not results:
            raise WatsonxIntegrationError(
                f"watsonx.ai response contained no 'results': "
                f"{list(response_json.keys())}"
            )

        first_result = results[0]
        generated_text = str(first_result.get("generated_text", "")).strip()
        if not generated_text:
            raise WatsonxIntegrationError(
                "watsonx.ai returned an empty generated_text."
            )

        logger.info(
            "Granite generation succeeded: model=%s, %d chars, %.2fs",
            self._config.model_id,
            len(generated_text),
            latency,
        )

        return GenerationResult(
            text=generated_text,
            model_id=str(response_json.get("model_id", self._config.model_id)),
            input_token_count=first_result.get("input_token_count"),
            generated_token_count=first_result.get("generated_token_count"),
            stop_reason=first_result.get("stop_reason"),
            latency_seconds=round(latency, 3),
        )

    def generate_safe(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[list] = None,
    ) -> Optional[GenerationResult]:
        """Convenience wrapper around :meth:`generate` that catches
        ``WatsonxIntegrationError`` and returns ``None`` instead of
        raising, for callers that want a single "try live generation,
        else fall back" branch without a try/except block of their own.

        Args:
            prompt: See :meth:`generate`.
            max_new_tokens: See :meth:`generate`.
            temperature: See :meth:`generate`.
            stop_sequences: See :meth:`generate`.

        Returns:
            A :class:`GenerationResult` on success, or ``None`` if
            generation failed for any reason (the failure is logged at
            WARNING level with the reason).
        """
        try:
            return self.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_sequences=stop_sequences,
            )
        except WatsonxIntegrationError as exc:
            logger.warning(
                "Granite generation unavailable, caller should fall back "
                "to a templated narrative. Reason: %s",
                exc,
            )
            return None
