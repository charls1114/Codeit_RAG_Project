// 채팅 전송 함수
async function sendChat() {
    const inputField = document.getElementById("chatInput");
    const chatBox = document.getElementById("chatBox");
    const sendBtn = document.getElementById("sendBtn");

    const question = inputField.value.trim();

    // 1. 빈칸이면 전송 안 함
    if (!question) return;

    // 2. 내 질문 화면에 표시
    addMessage(question, "user");
    inputField.value = ""; // 입력창 비우기
    inputField.disabled = true; // 전송 중 입력 방지
    sendBtn.disabled = true;

    // 3. 로딩 표시 ("생각 중...")
    const loadingId = addMessage("AI가 문서를 읽고 있어요...", "bot", true);

    try {
        // 4. FastAPI 서버로 요청 보내기
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            // 중요: 백엔드의 QueryRequest 모델(query)과 이름을 맞춰야 함
            body: JSON.stringify({ query: question })
        });

        if (!response.ok) throw new Error("서버 오류가 발생했습니다.");

        const data = await response.json();

        // 5. 로딩 메시지 지우고 실제 답변 표시
        removeMessage(loadingId);
        addMessage(data.answer, "bot");

    } catch (error) {
        console.error("Error:", error);
        removeMessage(loadingId);
        addMessage("죄송합니다. 오류가 발생했습니다 ㅠㅠ", "bot");
    } finally {
        inputField.disabled = false;
        sendBtn.disabled = false;
        inputField.focus();
    }
}

// 화면에 말풍선 추가하는 도우미 함수
function addMessage(text, type, isLoading = false) {
    const chatBox = document.getElementById("chatBox");
    const div = document.createElement("div");

    div.classList.add("chat-message", type);
    if (isLoading) {
        div.classList.add("loading-dots");
        div.id = "loading-" + Date.now(); // 나중에 지우기 위해 ID 부여
    }
    div.innerText = text; // HTML 태그 방지 (보안)

    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight; // 스크롤 맨 아래로

    return div.id;
}

// 로딩 메시지 삭제 함수
function removeMessage(id) {
    const element = document.getElementById(id);
    if (element) element.remove();
}

// 엔터키 입력 시 전송 기능
document.getElementById("chatInput").addEventListener("keypress", function(e) {
    if (e.key === "Enter") sendChat();
});
