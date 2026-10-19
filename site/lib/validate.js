// Validation for design-partner applications, shared by the Pages Function and the Node tests.
export const LIMITS = { name: 100, email: 200, company: 120, role: 100, indexes: 1000, stack: 300 };
export const SIZES = ['1-49', '50-199', '200-999', '1000-4999', '5000+'];
export const TIMELINES = ['now', 'this-quarter', 'next-quarter', 'exploring'];

export function validate(body) {
  const clean = {};
  for (const [field, max] of Object.entries(LIMITS)) {
    const value = typeof body[field] === 'string' ? body[field].trim() : '';
    if (value.length > max) return { error: `${field} is too long` };
    clean[field] = value;
  }
  for (const field of ['name', 'email', 'company', 'role', 'indexes']) {
    if (!clean[field]) return { error: `Please fill in ${field}.` };
  }
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(clean.email)) return { error: 'Please enter a valid work email.' };
  if (!SIZES.includes(body.company_size)) return { error: 'Please choose a company size.' };
  if (!TIMELINES.includes(body.timeline)) return { error: 'Please choose a timeline.' };
  if (body.consent !== true) return { error: 'Please agree to be contacted about the program.' };
  return { value: { ...clean, company_size: body.company_size, timeline: body.timeline, wants_call: body.wants_call === true } };
}
