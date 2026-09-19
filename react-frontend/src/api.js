const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${BASE_URL}/upload-document/`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Failed to upload document');
  }

  return response.json();
}

export async function chatWithAgent(sessionId, query, enableWebSearch) {
  const payload = {
    session_id: sessionId,
    query: query,
    enable_web_search: enableWebSearch,
  };

  const response = await fetch(`${BASE_URL}/chat/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Failed to get response from agent');
  }

  return response.json();
}
