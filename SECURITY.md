# Security Policy

## Supported versions

Security fixes are released for the latest `0.x` version only.

## Reporting a vulnerability

Do not open a public issue. Report privately through GitHub:
**Security** tab → **Report a vulnerability**
(https://github.com/DYNAMOatGHUB/TinyAgent-SLM/security/advisories/new).

Include the version, a description of the problem, and steps or code to
reproduce it. We aim to acknowledge reports within 7 days and to agree a
disclosure date with you once a fix is ready.

## Scope

In scope: anything that lets model output or a user query make SLMRouter run
code other than a registered tool with validated arguments, bypass the
model-call limit, or leak prompts or file paths through error messages.

Known, documented limitations that are **not** vulnerabilities in SLMRouter:

- **Prompt injection.** A user query can persuade the model to call any
  registered tool. Tools that change state must enforce their own
  authorization.
- **Tool timeouts cannot stop a thread.** A timed-out tool keeps running in the
  background. Tools that call external systems must enforce their own
  timeouts.
- **No sandboxing.** SLMRouter controls which registered function runs and with
  what arguments, not what that function does.
