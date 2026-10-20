// Landing-page behavior: copy button, optional call/GitHub links, and the design-partner form.
const cfg = document.body.dataset;
const configured = (url) => url && !url.includes('REPLACE_ME');

if (configured(cfg.cal)) {
  document.querySelectorAll('[data-call-link]').forEach((a) => { a.href = cfg.cal; a.target = '_blank'; a.rel = 'noopener'; });
  document.querySelectorAll('[data-call]').forEach((el) => el.classList.remove('hidden'));
}
if (configured(cfg.github)) {
  document.querySelectorAll('[data-github-link]').forEach((a) => { a.href = cfg.github; a.classList.remove('hidden'); });
}

document.querySelectorAll('[data-copy]').forEach((button) => button.addEventListener('click', async () => {
  const label = button.querySelector('.copy-label');
  try {
    await navigator.clipboard.writeText(button.dataset.copy);
    label.textContent = 'Copied';
  } catch {
    label.textContent = 'Select and copy';
  }
  setTimeout(() => { label.textContent = 'Copy'; }, 2000);
}));

const form = document.getElementById('apply');
const msg = form?.querySelector('.msg');

form?.addEventListener('submit', async (event) => {
  event.preventDefault();
  msg.textContent = '';
  msg.className = 'msg';
  if (!form.reportValidity()) return;
  const data = Object.fromEntries(new FormData(form));
  const body = { ...data, wants_call: form.wants_call.checked, consent: form.consent.checked, turnstile: data['cf-turnstile-response'] || '' };
  delete body['cf-turnstile-response'];
  if (!body.turnstile) {
    msg.textContent = 'Please complete the verification above.';
    msg.classList.add('error');
    return;
  }
  const button = form.querySelector('button[type=submit]');
  button.disabled = true;
  button.textContent = 'Sending…';
  try {
    const res = await fetch('/api/apply', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
    const out = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(out.error || `Something went wrong (${res.status}).`);
    form.classList.add('hidden');
    const thanks = document.getElementById('thanks');
    thanks.classList.remove('hidden');
    thanks.focus();
  } catch (error) {
    msg.textContent = error.message;
    msg.classList.add('error');
    if (window.turnstile) window.turnstile.reset();
  } finally {
    button.disabled = false;
    button.textContent = 'Apply';
  }
});
