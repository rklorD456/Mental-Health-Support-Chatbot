let SESSION_ID = crypto.randomUUID();

function resetChat() {
    // Generate a fresh session ID
    SESSION_ID = crypto.randomUUID();
    const chatBox = document.getElementById("chat-box");

    // Reset the chat and recreate the typing indicator so it isn't lost
    chatBox.innerHTML = `
        <div class="message-wrapper wrapper-bot">
            <div class="message bot-message">Hello! I am here to listen and support you. How are you doing today?</div>
        </div>
        <div class="typing-indicator" id="typing-indicator" style="display: none;">
            <div class="dots">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
            </div>
        </div>
    `;

    document.getElementById("user-input").focus();
}

async function sendMessage() {
    const inputField = document.getElementById("user-input");
    const message = inputField.value.trim();
    const typingIndicator = document.getElementById("typing-indicator");
    const chatBox = document.getElementById("chat-box");

    if (!message) return;

    appendMessage(message, "user-message", "wrapper-user");
    inputField.value = "";

    // Show typing indicator
    if (typingIndicator) {
        chatBox.appendChild(typingIndicator);
        typingIndicator.style.display = "block";
    }
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message: message, session_id: SESSION_ID })
        });

        if (!response.ok) {
            throw new Error("Server response was not ok.");
        }

        const data = await response.json();

        if (typingIndicator) typingIndicator.style.display = "none";
        appendMessage(data.response, "bot-message", "wrapper-bot");

    } catch (error) {
        if (typingIndicator) typingIndicator.style.display = "none";
        appendMessage("I'm sorry, I am having trouble connecting to the server right now. Please try again in a moment.", "bot-message", "wrapper-bot");
    }
}

function appendMessage(text, msgClass, wrapperClass) {
    const chatBox = document.getElementById("chat-box");

    const wrapperDiv = document.createElement("div");
    wrapperDiv.className = `message-wrapper ${wrapperClass}`;

    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${msgClass}`;
    msgDiv.textContent = text;

    wrapperDiv.appendChild(msgDiv);
    chatBox.appendChild(wrapperDiv);

    // Keep typing indicator at the very bottom
    const typingIndicator = document.getElementById("typing-indicator");
    if (typingIndicator) {
        chatBox.appendChild(typingIndicator);
    }

    chatBox.scrollTop = chatBox.scrollHeight;
}

function handleKeyPress(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
}

window.onload = () => {
    document.getElementById("user-input").focus();
};