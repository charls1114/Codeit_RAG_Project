export const chatUI = {
    // 요소를 미리 저장하지 않고, getter로 필요할 때마다 가져옵니다 (안전성 확보)
    get chatBox() { return document.getElementById("chatBox"); },
    get inputField() { return document.getElementById("chatInput"); },
    get sendBtn() { return document.getElementById("sendBtn"); },



    appendMessage(text, sender, isLoading = false) {
        const wrapper = document.createElement("div");
        wrapper.className = `message-wrapper ${sender}`;

        // [수정] 이모티콘 대신 SVG 아이콘 코드 사용
        const icon = document.createElement("div");
        icon.className = "profile-icon";

        if (sender === "bot") {
            // 봇 아이콘 (챗봇 모양)
            icon.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                </svg>`;
        } else {
            // 유저 아이콘 (사람 모양)
            icon.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                    <circle cx="12" cy="7" r="4"></circle>
                </svg>`;
        }

        const content = document.createElement("div");
        content.className = "message-content";

        if (isLoading) {
            content.innerHTML = `<div class="loading-dots"><span></span><span></span><span></span></div>`;
            wrapper.id = "loading-msg";
        } else {
            // HTML 태그가 섞여 있어도 텍스트로 안전하게 렌더링 (보안상 textContent 추천하지만, 줄바꿈 위해 innerHTML 사용 시 주의)
            content.innerHTML = text.replace(/\n/g, "<br>");
        }

        wrapper.appendChild(icon);
        wrapper.appendChild(content);

        if (this.chatBox) {
            this.chatBox.appendChild(wrapper);
            this.scrollToBottom();
        }
    },


    removeLoading() {
        const loading = document.getElementById("loading-msg");
        if (loading) loading.remove();
    },

    scrollToBottom() {
        if (this.chatBox) {
            this.chatBox.scrollTop = this.chatBox.scrollHeight;
        }
    },

    toggleInput(disabled) {
        if (this.inputField && this.sendBtn) {
            this.inputField.disabled = disabled;
            this.sendBtn.disabled = disabled;
            if(!disabled) this.inputField.focus();
        }
    },

    getInput() {
        if (!this.inputField) return "";
        const val = this.inputField.value.trim();
        this.inputField.value = "";
        return val;
    }
};
