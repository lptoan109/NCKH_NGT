// File: static/recorder.js

async function sendAudioToServer(audioBlob) {
    recordingPanel.style.display = 'none';
    resultsPanel.style.display = 'block';
    resultsContent.innerHTML = '<p>Đang phân tích... Vui lòng chờ trong giây lát.</p>'; // Thông báo chờ
    resultPlayer.style.display = 'none';

    const formData = new FormData();
    formData.append('audio_data', audioBlob);

    // --- THÊM ĐOẠN CODE NÀY ---
    // Lấy theme hiện tại từ localStorage
    const currentTheme = localStorage.getItem('theme') || 'default';
    formData.append('theme', currentTheme);
    // --- KẾT THÚC PHẦN THÊM ---

    try {
        const response = await fetch('/upload_audio', { method: 'POST', body: formData });
        const data = await response.json();

        // ... phần còn lại của hàm giữ nguyên ...
        if (data.success) {
            let iconHtml = '';
            let resultClass = '';

            // Chọn icon và màu sắc dựa trên kết quả
            if (data.diagnosis_result === 'Khỏe mạnh') {
                iconHtml = '<i class="fas fa-check-circle"></i>';
                resultClass = 'success';
            } else {
                iconHtml = '<i class="fas fa-exclamation-triangle"></i>';
                resultClass = 'warning';
            }
            
            // Tạo giao diện kết quả mới
            const resultHtml = `
                <div class="result-display ${resultClass}">
                    <div class="result-icon">${iconHtml}</div>
                    <p class="result-text-main">${data.diagnosis_result}</p>
                    <p class="result-text-sub">Lưu ý: Kết quả chỉ mang tính chất tham khảo.</p>
                </div>
            `;
            resultsContent.innerHTML = resultHtml;
            audioPlayer.src = data.filename;
            resultPlayer.style.display = 'block';

        } else {
            resultsContent.innerHTML = '<h2>Đã có lỗi xảy ra</h2><p>Không thể phân tích file âm thanh.</p>';
        }
    } catch (error) {
        console.error('Error uploading audio:', error);
        resultsContent.innerHTML = '<h2>Đã có lỗi xảy ra</h2><p>Lỗi kết nối tới server.</p>';
    }
}