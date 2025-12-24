// 서버 통신 로직
export const chatAPI = {
    async sendMessage(query) {
        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: query })
            });

            if (!response.ok) throw new Error("Server Error");
            return await response.json();
        } catch (error) {
            console.error(error);
            throw error;
        }
    }
};
