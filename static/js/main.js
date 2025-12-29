import { chatAPI } from './api.js';
import { chatUI } from './ui.js';

async function handleSend() {
    const question = chatUI.getInput();
    if (!question) return;

    chatUI.appendMessage(question, "user");
    chatUI.toggleInput(true);
    chatUI.appendMessage("", "bot", true);

    try {
        const data = await chatAPI.sendMessage(question);
        chatUI.removeLoading();
        chatUI.appendMessage(data.answer, "bot");
    } catch (e) {
        chatUI.removeLoading();
        chatUI.appendMessage("오류가 발생했습니다.", "bot");
    } finally {
        chatUI.toggleInput(false);
    }
}

// ✅ 핵심 수정: DOMContentLoaded 안으로 이벤트 연결을 넣습니다.
document.addEventListener("DOMContentLoaded", () => {
    const sendBtn = document.getElementById("sendBtn");
    const chatInput = document.getElementById("chatInput");

    if (sendBtn && chatInput) {
        sendBtn.addEventListener("click", handleSend);
        chatInput.addEventListener("keypress", (e) => {
            if (e.key === "Enter") handleSend();
        });
        console.log("이벤트 연결 성공!"); // 콘솔에 이게 뜨는지 확인하세요
    } else {
        console.error("요소를 찾을 수 없습니다. HTML ID를 확인하세요: sendBtn, chatInput");
    }
});
