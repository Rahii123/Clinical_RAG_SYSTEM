const chatArea = document.getElementById('chat-area');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const sourcesList = document.getElementById('sources-list');
const validationBox = document.getElementById('validation-box');
const vText = document.getElementById('v-text');
const sourceModal = document.getElementById('source-modal');
const modalBody = document.getElementById('modal-body');
const closeModal = document.querySelector('.close-modal');

// Global store for the latest query's sources to enable clicks
let currentSources = [];

// Maintain a session ID for multi-user/conversational memory
const sessionId = 'session_' + Math.random().toString(36).substr(2, 9);

async function callRAGApi(query) {
    // 1. Add User Message
    addMessage(query, 'user');
    userInput.value = '';

    // 2. Show Loading State
    const botMsgContainer = addMessage('', 'bot', true);

    try {
        const response = await fetch('http://localhost:8000/api/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                session_id: sessionId
            })
        });

        if (!response.ok) throw new Error('API Failure');

        const data = await response.json();
        currentSources = data.sources; // Save sources for citation clicking

        // 3. Update Bot Message
        botMsgContainer.innerHTML = formatMarkdown(data.answer);

        // 4. Update Side Panel
        updateEvidence(data);

        // Auto-scroll to show citations
        chatArea.scrollTop = chatArea.scrollHeight;

    } catch (error) {
        console.error(error);
        botMsgContainer.innerHTML = `<span style="color:red">Failed to connect to Clinical RAG server. Make sure server.py is running.</span>`;
    }
}

function addMessage(text, type, isLoading = false) {
    const div = document.createElement('div');
    div.className = `message ${type}`;
    div.innerHTML = `<div class="msg-content">${isLoading ? '<div class="typing-indicator"><span></span><span></span><span></span></div>' : text}</div>`;
    chatArea.appendChild(div);
    chatArea.scrollTop = chatArea.scrollHeight;
    return div.querySelector('.msg-content');
}

function updateEvidence(data) {
    // Update Validation
    validationBox.classList.remove('unverified', 'verified');
    if (data.grounding === 'VERIFIED') {
        validationBox.classList.add('verified');
        vText.innerText = "All synthetic claims verified via NLI Grounding Layer.";
    } else {
        validationBox.classList.add('unverified');
        vText.innerText = "Warning: Some claims could not be verified by NLI.";
    }

    // Update Sources
    sourcesList.innerHTML = '<p class="section-title">Verified Sources</p>';
    data.sources.forEach(s => {
        const card = document.createElement('div');
        card.className = 'source-card';
        card.onclick = () => showSource(s.id);
        card.innerHTML = `
            <span class="s-id">[Source ${s.id}]</span>
            <span class="s-title">${s.title}</span>
            <span class="s-year">Section: ${s.section} • Year: ${s.year}</span>
        `;
        sourcesList.appendChild(card);
    });
}

function showSource(id) {
    const source = currentSources.find(s => s.id === id);
    if (source) {
        modalBody.innerText = source.content;
        sourceModal.style.display = 'block';
    }
}

function formatMarkdown(text) {
    return text
        // Headers
        .replace(/^### (.*$)/gim, '<h3>$1</h3>')
        .replace(/^## (.*$)/gim, '<h2>$1</h2>')
        .replace(/^# (.*$)/gim, '<h1>$1</h1>')
        // Bold
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        // List items
        .replace(/^\s*[-*•·]\s+(.*)$/gim, '<li>$1</li>')
        .replace(/(<li>.*<\/li>)/gms, '<ul>$1</ul>')
        // Fix duplicate <ul> tags
        .replace(/<\/ul>\s*<ul>/g, '')
        // Numbered lists
        .replace(/^\d+\.\s+(.*)$/gim, '<li class="num-list">$1</li>')
        // Newlines
        .replace(/\n\n/g, '<br><br>')
        .replace(/\n/g, '<br>')
        // Citations
        .replace(/\[Source\s+(\d+)\]/g, (match, id) => {
            return `<span class="citation-tag" onclick="showSource(${parseInt(id)})">[${id}]</span>`;
        });
}

function setQuery(query) {
    userInput.value = query;
    callRAGApi(query);
}

sendBtn.addEventListener('click', () => {
    if (userInput.value.trim()) callRAGApi(userInput.value);
});

userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && userInput.value.trim()) callRAGApi(userInput.value);
});

closeModal.onclick = () => sourceModal.style.display = 'none';
window.onclick = (e) => { if (e.target == sourceModal) sourceModal.style.display = 'none'; }
