document.addEventListener('DOMContentLoaded', () => {
    // ... (Phần code ghi âm từ đầu đến trước hàm sendAudioToServer giữ nguyên) ...
    const recordButton = document.getElementById('record-button');
    const timerElement = document.getElementById('timer');
    const recordingPanel = document.getElementById('recording-panel');
    const resultsPanel = document.getElementById('results-panel');
    const resultsContent = document.getElementById('results-content'); // Đổi sang vùng chứa mới
    const resultPlayer = document.getElementById('result-player');
    const audioPlayer = resultPlayer.querySelector('audio');
    const diagnoseAgainButton = document.getElementById('diagnose-again-button');

    if (!recordButton) return;

    let mediaRecorder;
    let audioChunks = [];
    let timerInterval;
    let seconds = 0;

    function updateTimer() {
        seconds++;
        const mins = String(Math.floor(seconds / 60)).padStart(2, '0');
        const secs = String(seconds % 60).padStart(2, '0');
        timerElement.textContent = `${mins}:${secs}`;
    }

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = event => { audioChunks.push(event.data); };
            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                sendAudioToServer(audioBlob);
                audioChunks = [];
            };
            mediaRecorder.start();
            recordButton.classList.add('is-recording');
            seconds = 0;
            timerElement.textContent = '00:00';
            timerInterval = setInterval(updateTimer, 1000);
        } catch (error) {
            console.error("Error accessing microphone:", error);
            alert("Không thể truy cập micro. Vui lòng cấp quyền và thử lại.");
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
            recordButton.classList.remove('is-recording');
            clearInterval(timerInterval);
        }
    }

    async function sendAudioToServer(audioBlob) {
        recordingPanel.style.display = 'none';
        resultsPanel.style.display = 'block';
        resultsContent.innerHTML = '<p>Đang phân tích... Vui lòng chờ trong giây lát.</p>'; // Thông báo chờ
        resultPlayer.style.display = 'none';

        const formData = new FormData();
        formData.append('audio_data', audioBlob);

        try {
            const response = await fetch('/upload_audio', { method: 'POST', body: formData });
            const data = await response.json();

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
    
    recordButton.addEventListener('click', () => {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            stopRecording();
        } else {
            startRecording();
        }
    });

    diagnoseAgainButton.addEventListener('click', () => {
        resultsPanel.style.display = 'none';
        recordingPanel.style.display = 'block';
        seconds = 0;
        timerElement.textContent = '00:00';
    });
});