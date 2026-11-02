# Security policy

RAG Shield is a security tool, so reports about it are welcome and taken seriously.

## Reporting a vulnerability

Email **hello@ragshield.dev** with "security" in the subject. Please include what you found, how to reproduce it, and the
version (`python -c "import ragshield; print(ragshield.__version__)"`). Please do not open a public issue for a
vulnerability until it is fixed.

You can expect an acknowledgement within 3 business days and a status update within 10. Fixes are released as a new
version and credited to the reporter unless you prefer otherwise.

## In scope

- Ways to get a tampered, unsigned or quarantined chunk served as verified
- Ways to bypass the trust filter, delta scoring or the hidden-character check that the README does not already list
  under Limits
- Leaks of document text from the ledger, the exposure report or logs

Detection misses by the injection classifier on new attack wording are expected and documented; they are useful
benchmark contributions rather than vulnerabilities, and an issue is the right place for them.

## Supported versions

Only the latest release receives fixes while the project is in 0.x.
