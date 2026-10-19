import assert from 'node:assert/strict';
import { test } from 'node:test';
import { validate } from '../lib/validate.js';

const good = {
  name: 'Ada', email: 'ada@example.com', company: 'Example Co', role: 'AI platform lead', company_size: '200-999',
  indexes: 'Confluence and Jira', stack: 'LangChain + pgvector', timeline: 'this-quarter', consent: true, wants_call: true,
};

test('accepts a complete application and trims fields', () => {
  const { value, error } = validate({ ...good, name: '  Ada  ' });
  assert.equal(error, undefined);
  assert.equal(value.name, 'Ada');
  assert.equal(value.wants_call, true);
});

test('requires consent, a valid email and known choices', () => {
  assert.match(validate({ ...good, consent: false }).error, /agree/);
  assert.match(validate({ ...good, email: 'not-an-email' }).error, /email/);
  assert.match(validate({ ...good, company_size: 'huge' }).error, /size/);
  assert.match(validate({ ...good, timeline: 'whenever' }).error, /timeline/);
  assert.match(validate({ ...good, indexes: '   ' }).error, /indexes/);
});

test('rejects oversized and non-string input', () => {
  assert.match(validate({ ...good, indexes: 'x'.repeat(1001) }).error, /too long/);
  assert.match(validate({ ...good, name: { $gt: '' } }).error, /name/);
  assert.equal(validate({ ...good, consent: 'true' }).error, 'Please agree to be contacted about the program.');
});
