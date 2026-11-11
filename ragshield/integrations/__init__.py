"""Drop-in retrievers for LangChain and LlamaIndex. Each returns only what the shield lets through.

The trust decision travels in each result's metadata: ragshield_action is 'pass' or 'warn', and ragshield_reason says why.
What to do with a warned chunk (show it, label it for the model, or leave it out) stays the application's choice."""


def trust_metadata(hit):
    trust = hit['trust']
    return {**hit.get('metadata', {}), 'chunk_id': hit['chunk_id'], 'source_id': trust.get('source_id'), 'trust_tier': trust.get('trust_tier'),
            'ragshield_action': trust['action'], 'ragshield_reason': trust['reason'], 'ragshield_trust': trust['trust'],
            'similarity': round(hit['score'], 4)}
