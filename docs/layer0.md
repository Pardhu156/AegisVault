# Layer 0 Deterministic Validation

Layer 0 is a fast rule-based validation layer that runs before expensive semantic checks.

It is deterministic and does not use:

- LLMs
- embeddings
- cosine similarity
- semantic classification
- external network calls

Layer 0 is disabled by default for backward compatibility with existing 4.2 policies.

## Pipeline Placement

Request checkpoint:

```text
incoming request
  -> Layer 0 request validation
  -> Request Gate
  -> protected application
```

Tool-call checkpoint:

```text
proposed tool call
  -> Layer 0 tool validation
  -> future Sentinel hook
  -> Action Gate
  -> tool execution
```

Layer 0 is not the final authorization authority. It handles structural validation and obvious deterministic policy violations. Request Gate, Action Gate, Goal Vault, Sentinel, and Response Gate retain their separate responsibilities.

## Policy Configuration

Existing policies without `layer0` continue to load and keep Layer 0 disabled.

```yaml
layer0:
  enabled: true
  fail_mode: closed
  stop_on_first_block: false

  request:
    require_session_id: true
    require_domain: true
    allowed_domains:
      - email
    max_characters: 20000
    max_bytes: 50000
    reserved_metadata_keys:
      - trusted_goal
      - goal_embedding
      - policy_internal
      - middleware_decision
      - sentinel_state
      - ema_drift
      - authorization_result
    forbidden_patterns:
      literals: []
      regex: []

  tools:
    allowlist_mode: true
    allowed:
      - read_email
      - summarize_email
    denied:
      - delete_email
    max_argument_bytes: 50000
    reserved_argument_keys:
      - trusted_goal
      - goal_embedding
      - sentinel_state
      - authorization
      - policy_override
      - middleware_context
    sensitive_argument_keys:
      - password
      - passwd
      - secret
      - api_key
      - access_token
      - refresh_token
      - private_key
    sensitive_argument_action: warn
    destination_rules:
      send_email:
        fields:
          - to
          - recipients
        allowed_values:
          - manager@example.com
```

## Request Rules

- `L0_REQUEST_TYPE_INVALID`
- `L0_REQUEST_EMPTY`
- `L0_REQUEST_TOO_LARGE`
- `L0_SESSION_MISSING`
- `L0_POLICY_MISSING`
- `L0_DOMAIN_MISSING`
- `L0_DOMAIN_NOT_ALLOWED`
- `L0_RESERVED_METADATA_KEY`
- `L0_GOAL_OVERWRITE_ATTEMPT`
- `L0_FORBIDDEN_PATTERN`

Forbidden patterns are configured by trusted developers. Layer 0 does not include a broad universal jailbreak keyword list.

## Tool Rules

- `L0_TOOL_NAME_MISSING`
- `L0_TOOL_UNDECLARED`
- `L0_TOOL_DENIED`
- `L0_TOOL_NOT_ALLOWED`
- `L0_TOOL_ARGUMENTS_INVALID`
- `L0_TOOL_SCHEMA_INVALID`
- `L0_TOOL_ARGUMENT_TOO_LARGE`
- `L0_TOOL_RESERVED_ARGUMENT`
- `L0_TOOL_SECRET_EXPOSURE`
- `L0_TOOL_EXTERNAL_DESTINATION`

Tool arguments are treated as untrusted data and are never mutated.

## Fail Modes

`fail_mode: closed` blocks on unexpected Layer 0 internal errors.

`fail_mode: open` allows continuation with a warning on unexpected Layer 0 internal errors.

Known policy violations still block even when fail mode is open.

## Audit Redaction

Layer 0 audit events include safe metadata such as checkpoint, decision, risk level, matched rule IDs, tool name, and validation latency.

Sensitive keys are redacted recursively and case-insensitively. Raw passwords, API keys, tokens, and private keys must not be logged.

## Direct Usage

```python
from aegisvault.layer0 import Layer0Validator

validator = Layer0Validator(policy=policy)

request_decision = validator.validate_request(
    session_id="session-123",
    request_text="Summarize my unread emails",
    domain="email",
)

tool_decision = validator.validate_tool_call(
    session_id="session-123",
    tool_name="send_email",
    arguments={"to": "manager@example.com", "body": "Draft"},
    domain="email",
)
```

## Latency Check

Run the deterministic micro-benchmark:

```bash
python evaluation/layer0/benchmark_layer0.py
```

This measures local validation overhead only. It does not call Ollama or any external service.
